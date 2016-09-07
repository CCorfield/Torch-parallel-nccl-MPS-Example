--------------------------------------------------------------------------------
-- This module provides a training harness to train a network.
-- It provides support for a multi-GPU, multi-Process environment.
-- By setting numWorkersPerGPU you can control the total number of training 
-- processes, which is numGPUs * numWorkersPerGPU.
-- We use Torch/parallel to fork and exec child processes and manage 
-- inter-process communication.
-- In order that each process has an identical copy of the network, we send the 
-- parameters of the parent's network to all the children at start-up.
-- Likewise, when updating the network, we use nccl/AllReduce to exchange and
-- sum the gradient parameters between all the processes. Note that we only
-- transmit the accumulated gradient data and not the new parameters. This saves
-- on time and bandwidth.
-- Note also the use of getParameters(), called once (for each worker) at start-up
-- to get persistent pointers to the parameters and gradient parameters of each
-- network. See how we use gradParameters with nccl/AllReduce to share
-- gradient data between all the networks.
-- If you set "debug = true" (scroll down a couple of screens), you can get
-- copious output from the training harness.
--------------------------------------------------------------------------------

-- Required libraries
require 'cutorch'
require 'optim' 
require 'parallel'
require 'sys'
local ffi = require 'ffi'
local nccl = {}
nccl.C = require 'nccl_ffi'
local Network = require 'Network'
local DataSets = require 'DataSets'  
require 'ccNesterov'

-- Which GPU the parent runs on (do not change)
local parentGPU = 1

-- Saving and loading models
local saveFileName = "ExampleModel.t7"
local loadFileName = nil -- If true, load model from file, else initialize fresh model

-- Model and training parameters, and common cases are enumerated
-- Refer to Torch/optim for SGD, AdaGrad, and AdaDelta, and others
-- ccNesterov is home grown
local modelConfig = {} -- Kitchen sink for whatever values you want to pass along
modelConfig.method = ccNesterov     -- optim.sgd, optim.adadelta, optim.adagrad, ccNesterov
modelConfig.learningRate = 1e-3     -- SGD, AdaGrad, ccNesterov
modelConfig.learningRateDecay = nil -- SGD, AdaGrad
modelConfig.learningRates = nil     -- SGD
modelConfig.nesterov = nil          -- SGD
modelConfig.momentum = 0.9          -- SGD, ccNesterov [0.9] 
modelConfig.dampening = nil         -- SGD [0.0] 
modelConfig.rho = nil               -- AdaDelta [0.9] 
modelConfig.eps = nil               -- AdaDelta [1e-6]
modelConfig.weightDecay = nil       -- SGD, AdaDelta, AdaGrad
modelConfig.weightDecays = nil      -- SGD, AdaDelta, AdaGrad

-- How many parallel training processes there will be:
local numGPUs = 2
local numWorkersPerGPU = 2
local numWorkers = numWorkersPerGPU * numGPUs

-- Epochs, validation, and termination
local numEpochs = 1000
local numSamplesPerUpdate = 1        -- When to update model
local numUpdatesPerValidation = 10   -- When to evaluate current accuracy
local numEpochsPerModelSave = 100    -- Periodic saving off of models
local numUpdatesPerEpoch = nil       -- If nil, value will be calculated (preferred)
local validationErrorRate = 1.0      -- Initial value, before we do the first validation
local validationThreshold = 0.001    -- Used for early termination
local validationThresholdCount = 0   -- Used for early termination
local validationThresholdLimit = 100 -- Stop after Nth time validation error rate meets threshold
local terminate = false

local debug = false

-- NCCL communicator object
local ncclComm, ncclResult, ncclErrString

-- Parent process:
function parentProcess()

  local model
  local replies
  
  -- Logger initialization
  local logger = optim.Logger('ExampleModel.log')
  logger:setNames({'Average Loss Per Sample', 'Average Validation Error %'})
  logger:style({'-', '-'})

  -- Create a reference model
  cutorch.setDevice(parentGPU)
  model = Network.initModel(parentGPU, nil, true, modelConfig)
  parallel.print("Number of model parameters = " .. Network.getParams():size(1))
   
  -- Fork and exec all the workers in one go
  parallel.nfork(numWorkers - 1)
  parallel.children:exec(workerProcess) 
  
  -- Set up GPU assignments
  local assignments = {}
  local childId = 0 
  for gpu = 1, numGPUs do 
    local maxNumWorkerPerGPU = numWorkersPerGPU
    if (gpu == 1) then
      maxNumWorkerPerGPU = maxNumWorkerPerGPU - 1 -- Parent counts as one worker
    end
    for worker = 1, maxNumWorkerPerGPU do
      childId = childId + 1
      local childGPU = gpu 
      table.insert(assignments, {gpu = childGPU, childID = childId, pid = parallel.children[childId].unixid})
    end
  end
  
  -- Send GPU assignments:
  parallel.children:join()
  parallel.children:send(assignments)
  replies = parallel.children:receive() -- Use for debugging
  replies = nil -- Recycle memory
 
  -- Send model's configuration parameters:
  parallel.children:join()
  parallel.children:send(modelConfig)
  replies = parallel.children:receive() -- Use for debugging
  replies = nil -- Recycle memory
 
  -- Send num jobs per update:
  parallel.children:join()
  parallel.children:send(numSamplesPerUpdate)
  replies = parallel.children:receive() -- Use for debugging
  replies = nil
 
 -- Send model's initial parameters:
 parallel.children:join()
 parallel.children:send(Network.getParams())
 replies = parallel.children:receive()
 replies = nil
 if (debug) then -- Sanity check that everyone has the same parameters
   parallel.print("params norm = " .. Network.getParamsNorm())
 end
 
 -- Create and send uniqueId for nccl communicator clique:
  local uniqueId = ffi.new('ncclUniqueId')
  nccl.C.ncclGetUniqueId(uniqueId)
  local x = torch.zeros(128):int()
  for i = 0,127 do
    x[i+1] =  uniqueId.internal[i]
  end
  if(debug) then -- Sanity check that everyone has the same UniqueID for nccl communications
    local s = ""
    for i = 0, 127 do
      if uniqueId.internal[i] == 0 then break end
      s = s .. " " .. uniqueId.internal[i]
    end
    parallel.print("UniqueId = " .. s)
  end 
  parallel.children:join()
  parallel.children:send(x)

  -- Init "rank" and "communicator" for parent process:
  ncclComm = ffi.new('ncclComm_t[?]',1)
  ncclResult = nccl.C.ncclCommInitRank(ncclComm, numWorkers, uniqueId, parallel.id)
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  assert(ncclResult == nccl.C.ncclSuccess, ncclErrString)
  if(debug) then
    parallel.print("nccl communicator initialized with result = " .. ncclErrString)
  end

  cutorch.setDevice(parentGPU) 
  
 -- Calculate # of updates per Epoch, if not otherwise defined:
 if(numUpdatesPerEpoch == nil) then
   numUpdatesPerEpoch = torch.round(DataSets.getNumberOfTrainingSamples() / 
                                    (numSamplesPerUpdate * numWorkers))
 end
 
  -- Utility functions for training loop:
  local updateCounter = 0
  local function updateValidationError(numUpdatesPerValidation)
    updateCounter = updateCounter + 1
    return (updateCounter%numUpdatesPerValidation == 0)
  end
  
  -- Various statistics to track:
  local totalLossPerEpoch = 0
  local totalNumSamplesPerEpoch = 0
  local averageLossPerSamplePerEpoch = 0
  local totalValidationErrorPerEpoch = 0
  local totalNumValidationsPerEpoch = 0
  local averageValidationErrorPerEpoch = 0
  local lastValidationError = 0

  -- Training loop:
  local lastEpoch = 0 -- Remembers the last epoch when we terminate the for-loop
  for epoch = 1, numEpochs do  
    print(string.format("Epoch %d/%d:",epoch, numEpochs))

    -- Reset statistics
    totalLossPerEpoch = 0
    totalNumSamplesPerEpoch = 0
    averageLossPerSamplePerEpoch = 0
    totalValidationErrorPerEpoch = 0
    totalNumValidationsPerEpoch = 0
    averageValidationErrorPerEpoch = 0
    
    -- Inner loop for pushing training data through network and updating parameters
    for update = 1, numUpdatesPerEpoch do
      
      -- Sync with children
      parallel.children:join('continue')

      Network.trainAndUpdateModel(numSamplesPerUpdate, numWorkers, ncclComm, parallel.id)

      -- Receive stats from children
      parallel.children:join()
      local childStats = parallel.children:receive()

     -- Get own statistics
     local parentStats = Network.getStatistics()

      -- Update loss statistics
      local totalLossPerUpdate = parentStats.Loss
      local totalNumSamplesPerUpdate = parentStats.Samples
      for i = 1, #childStats do
        totalLossPerUpdate = totalLossPerUpdate + childStats[i].Loss
        totalNumSamplesPerUpdate = totalNumSamplesPerUpdate + childStats[i].Samples
      end
      local averageLossPerSamplePerUpdate = totalLossPerUpdate / totalNumSamplesPerUpdate
      totalLossPerEpoch = totalLossPerEpoch + totalLossPerUpdate
      totalNumSamplesPerEpoch = totalNumSamplesPerEpoch + totalNumSamplesPerUpdate
           
      -- Update validation statistics
      local recalcValidationError = updateValidationError(numUpdatesPerValidation)
      if(recalcValidationError) then
        model:evaluate()
        cutorch.setDevice(model.gpu)
        validationErrorRate = Network.validateModel(model)
        totalValidationErrorPerEpoch = totalValidationErrorPerEpoch + validationErrorRate
        totalNumValidationsPerEpoch = totalNumValidationsPerEpoch + 1
        model:training()
      end
         
      -- Report statistics --
      
      -- Average Loss and validation error per update:
      print(string.format("Epoch %d/%d, Update %d/%d", epoch, numEpochs, update, numUpdatesPerEpoch))
      
      -- Average Loss and (last calculated) validation per epoch:
      if(update == numUpdatesPerEpoch) then -- Record at the end of an epoch
        averageLossPerSamplePerEpoch = totalLossPerEpoch / totalNumSamplesPerEpoch
        if(totalNumValidationsPerEpoch>0) then
          averageValidationErrorPerEpoch = totalValidationErrorPerEpoch/totalNumValidationsPerEpoch
          lastValidationError = averageValidationErrorPerEpoch
        else
          averageValidationErrorPerEpoch = lastValidationError -- Reuse last calculated value to smooth stats
        end
        print(string.format("Epoch %d/%d: Average loss per sample = %.2f, Average validation error = %.0f%%", 
              epoch, numEpochs, averageLossPerSamplePerEpoch, 100*averageValidationErrorPerEpoch))

        logger:add({averageLossPerSamplePerEpoch, averageValidationErrorPerEpoch*100})
      end

      -- Check for early termination based on current validation error rate:
      if(recalcValidationError) then -- We have just updated the error rate (see above)
        if (validationErrorRate <= validationThreshold) then
          validationThresholdCount = validationThresholdCount + 1
          -- If we have met the threhold N times, it is time to terminate training
          if(validationThresholdCount >= validationThresholdLimit) then
            terminate = true
          end
        end
      end
      
      -- Periodic memory recovery
      collectgarbage()
      
      if (terminate) then break end
    end
    
    -- Save interim model
    if(epoch%numEpochsPerModelSave  == 0) then
      torch.save(saveFileName .. "_epoch_" .. epoch, model)    
    end
    lastEpoch = epoch -- Remember the last epoch
    
    if (terminate) then break end
  end
  -- Out of training loop --
  
  -- Log final statistics:
  averageLossPerSamplePerEpoch = totalLossPerEpoch / totalNumSamplesPerEpoch
  if(totalNumValidationsPerEpoch>0) then
    averageValidationErrorPerEpoch = totalValidationErrorPerEpoch/totalNumValidationsPerEpoch
  else
    averageValidationErrorPerEpoch = lastValidationError -- Reuse last calculated value to smooth output
  end
  print(string.format("Epoch %d/%d: Average loss per sample = %.2f, Average validation error = %.0f%%",
        lastEpoch, numEpochs, averageLossPerSamplePerEpoch, 100*averageValidationErrorPerEpoch))
  logger:add({averageLossPerSamplePerEpoch, averageValidationErrorPerEpoch*100})
  
  -- Sync and terminate workers
  parallel.children:join('break')
  parallel.print('all processes terminated')
  
  -- Tear down nccl framework
  nccl.C.ncclCommDestroy(ncclComm[0])

  -- Save Model
  torch.save(saveFileName, model)
  
  -- Plot training graph
  logger:plot()
end

-- Worker processes:
function workerProcess()
  -- A worker starts with a blank environment, which means all the libraries
  -- must be re-loaded, since they won't be in scope after the parent's fork.
  require 'cutorch'
  require 'sys'
  require 'parallel'
  local ffi = require 'ffi'
  local nccl = {}
  nccl.C = require 'nccl_ffi'
  local Network = require 'Network'

  local model
  local newparams = nil
  local myGPU, myPID
  local debug = false
  
  -- Receive GPU and PID assignments:
  parallel.yield()
  local assignments = parallel.parent:receive()
  numWorkers = #assignments + 1 -- Add one for parent process
  parallel.parent:send('Assignments -- ok')
  for i=1,#assignments do
    local gpu = assignments[i].gpu
    local childID = assignments[i].childID
    local pid = assignments[i].pid
    if parallel.id == childID then
      myGPU = gpu
      myPID = pid
      if(debug) then
        parallel.print("GPU #" .. myGPU)
        parallel.print("Unix pid " .. myPID)
        io.flush()
      end
    end
  end
  
  -- Set up model and training method:
  parallel.yield()
  local modelConfig = parallel.parent:receive()
  parallel.parent:send('model configuration -- ok')
  model = Network.initModel(myGPU, nil, true, modelConfig)

  -- Receive number of samples per model update:
  parallel.yield()
  local numSamplesPerUpdate = parallel.parent:receive() 
  parallel.parent:send('numSamplesPerUpdate -- ok')
  if(debug) then
    parallel.print("Number of Samples per update = " .. numSamplesPerUpdate)
    io.flush()
  end
    
  -- Re-initialize RNGs
  math.randomseed(myPID)
  torch.manualSeed(myPID) 
  cutorch.setDevice(myGPU)
  cutorch.manualSeed(myPID)

  -- Receive initial parameters:
  parallel.yield()
  newparams = parallel.parent:receive()
  parallel.parent:send('params -- ok')
  Network.setParamValues(newparams)
  newparams = nil
  if(debug) then -- Sanity check that child parameters makes sense
    parallel.print("params norm = " .. Network.getParamsNorm())
    io.flush()
  end
  
  -- Set up nccl interface --
  
  -- Receive uniqueId for communicator clique:
  local uniqueId = ffi.new('ncclUniqueId')
  parallel.yield()
  local x = parallel.parent:receive()
  local s = ""
  for i = 0,127 do
    uniqueId.internal[i] = x[i+1]
    if x[i+1] ~= 0 then s = s .. " " .. x[i+1] end
  end
  if(debug) then
    parallel.print("UniqueId = " .. s)
    io.flush()
  end
  
  -- Init this worker's "rank" and "communicator":
  cutorch.setDevice(myGPU)
  ncclComm = ffi.new('ncclComm_t[?]',1)
  ncclResult = nccl.C.ncclCommInitRank(ncclComm, numWorkers, uniqueId, parallel.id)
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  assert(ncclResult == nccl.C.ncclSuccess, ncclErrString)
  if(debug) then
    parallel.print("nccl communicator initialized with result = " .. ncclErrString)
    io.flush()  
  end

  cutorch.setDevice(myGPU)
   
  -- Working loop and termination
  while true do
    -- Yield to parent for termination
    local m = parallel.yield()
    if m == 'break' then break end
    if(debug) then
      parallel.print("Message at top of while loop = " .. m)
      io.flush()
    end
    
    -- Train and update model
    Network.trainAndUpdateModel(numSamplesPerUpdate, numWorkers, ncclComm, parallel.id)
    
    -- Send statistics
    local stats = Network.getStatistics()
    parallel.yield()
    parallel.parent:send(stats)
    
    collectgarbage()
  end
  
  -- Tear down nccl framework
  nccl.C.ncclCommDestroy(ncclComm[0])
  
  collectgarbage()
end

-- protected execution:
local ok,err = pcall(parentProcess)
if not ok then print(err) parallel.close() end
