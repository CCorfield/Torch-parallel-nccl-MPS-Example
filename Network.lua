----------------------------------------------------------------------------------
-- This module provides a wrapper for initializing, training and updating a network.
-- Models can be initialized from scratch, loaded from storage, and set to either 
-- training or evaluation modes.
-- You can choose from a variety of training techniques, which are offered in the
-- Torch/optim library, plus a homegrown version of Nesterov accelerated convergence.
-- As more algorithms become available in Torch/optim you can easily plug them in
-- by following the current examples.
-- We use nVidia's "nccl" library to transfer data directly between model-clones
-- stored on the GPUs. The transfer times will depend on the bus/memory architecture
-- of the node you are running on.
-- If you set "debug = true", you will get additional output.
----------------------------------------------------------------------------------


require 'optim'
require 'nnx'
require 'parallel'
require 'sys'
local ffi = require 'ffi'
local nccl = {}
nccl.C = require 'nccl_ffi'

local Model = require 'Model' 
local DataSets = require 'DataSets'
local Evaluate = require 'Evaluate'

local Network = {}

local debug = false
local model, params, gradParams

-- These are the items used by the various update algorithms defined in the optim package
local updateConfig = {} -- See below to set the values you want
local updateMethodState = {} -- Used to cache values between updates

-- Initialize the modeling and training parameters
-- Return model
function Network.initModel(gpu, loadFileName, trainModel, modelConfig)

  if (gpu) then -- Load gpu modules.
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
  end

  if(gpu) then 
    cutorch.setDevice(gpu)
  end
  
  -- Load or assign model
  modelConfig = modelConfig or {} -- If we're not training leave as an empty table
  if (loadFileName ~= nil) then
    model = torch.load(loadFileName)
  else
    model = Model.init(modelConfig)
  end
  
  -- Set up model
  if(gpu) then
    cutorch.setDevice(gpu)
    model = model:cuda() 
    model.gpu = gpu
    model.inputBuffer = torch.CudaTensor()
  else
    model.inputBuffer = torch.Tensor()
  end
  
  -- Set training/evaluation mode
  if(trainModel) then 
    model:training()
  else
    model:evaluate()
  end

  -- Initialize (persistent) params and gradParams
  params, gradParams = model:getParameters()
  
  -- Set up the update method
  updateConfig = modelConfig
  
  -- Return model
  return model
end

function Network.getParams()
  return params
end

function Network.setParamValues(paramValues)
  params:copy(paramValues)
end

function Network.getGradParams()
  return gradParams
end

function Network.getParamsNorm()
  return torch.mean(torch.abs(params))
end

function Network.getGradParamsNorm()
  return torch.mean(torch.abs(gradParams))
end

function Network.getParamsRMS()
  return params:norm(2)/math.sqrt(params:size(1))
end

function Network.getGradParamsRMS()
  return gradParams:norm(2)/math.sqrt(gradParams:size(1))
end

-- Statistics
local totalLoss = 0
local totalSampleCount = 0
local sampleLoadTime = 0
local gpuFwdBwdTime = 0
  
function Network.reInitStatistics()
  totalLoss = 0
  totalSampleCount = 0
  sampleLoadTime = 0
  gpuFwdBwdTime = 0
end

function Network.getStatistics()
  return {Loss = totalLoss, Samples = totalSampleCount, 
          SampleLoadTime = sampleLoadTime, GPUFwdBwdTime = gpuFwdBwdTime}
end

-- Training
local function parallelSyncAll(msg)
  local replies
  if(parallel.id == 0) then
    parallel.children:join()
    if(debug) then
      parallel.children:send(msg)
      parallel.print("Message to children: " .. msg)
      io.flush()
      replies = parallel.children:receive()
      parallel.print("# replies = " .. #replies)
      io.flush()
    end
  else 
    parallel.yield()
    if(debug) then
      local message = parallel.parent:receive()
      parallel.print("Message from parent: " .. message)
      io.flush()
      parallel.parent:send("Received message = " .. message)
    end
  end
end

-- Initialize parameters used to calculate learning rates and updates
local runningGradParamsNorm  = 0
local runningDeltaParamsNorm = 0
local rho = 0.9 -- Factor used in running averages
local epsilon = 1e-6 -- Prevents underflows
local learningRate = 0
local deltaParamsNorm = 0
local paramsNorm = 0
local gradParamsNorm = 0

function Network.trainAndUpdateModel(numSamples, numWorkers, comm, workerID)
  local ncclResult, ncclErrString
  
  if(debug) then
    parallelSyncAll("Top of Network.trainAndUpdateModel()")
  end
  
  -- Reset statistics
  Network.reInitStatistics()

  -- Push training data through model
  gradParams:zero()
  
  -- Forward/Backward
  for b = 1, numSamples do
    local predictions, gradOutput, samples
    
    -- Forward and backward on training sample
    cutorch.setDevice(model.gpu)
    
    sys.tic() -- Sample Load Time
    samples = DataSets.getTrainingSamples()
    sampleLoadTime = sampleLoadTime + sys.toc()
    
    sys.tic() -- GPU Fwd/Bwd Time
    -- Note the use of "typeAs()" to ensure compatible data types in the relevant operation
    model.inputBuffer:resize(samples.inputs:size()):copy(samples.inputs)  
    predictions = model:forward(model.inputBuffer)
    totalLoss  = totalLoss + model.criterion:forward(predictions, samples.targets:typeAs(predictions))
    gradOutput = model.criterion:backward(predictions, samples.targets:typeAs(predictions)) 
    model:backward(model.inputBuffer, gradOutput)
    totalSampleCount = totalSampleCount + samples.inputs:size(1)
    gpuFwdBwdTime = gpuFwdBwdTime + sys.toc()
    
    -- Mark memory for recovery
    predictions = nil
    gradOutput = nil
    samples = nil
  end
  
  parallelSyncAll("Just after forward/backward")

  cutorch.setDevice(model.gpu)
  sys.tic()
  collectgarbage('stop')
  assert(gradParams:isContiguous(), "gradParams not contiguous") 
  parallelSyncAll("Just before AllReduce")
  ncclResult = nccl.C.ncclAllReduce(gradParams:data(), gradParams:data(), -- In place version
    gradParams:size(1), nccl.C.ncclFloat,
    nccl.C.ncclSum, comm[0],
    ffi.C.THCState_getCurrentStream(cutorch.getState()))
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  assert(ncclResult == nccl.C.ncclSuccess, ncclErrString) 
  assert(cutorch.getDevice() == model.gpu, "Mismatch in current GPU") -- There shouldn't be any change in device through the AllReduce() 
  parallelSyncAll("Just after AllReduce. Status = " .. ncclErrString)
    
  if(debug or parallel.id == 0) then
    parallel.print("GradParams norm = " .. Network.getGradParamsNorm())
    io.flush()
  end  

  collectgarbage('restart')
  
  -- Prevent numerical instabilities:
  -- gradParams:clamp(-val, val) Figure out values to put here, if necessary

  -- Here is where we do the model update according to the selected method (see above)
  -- Note the implicit assumption that everything is first order (i.e. linear)
  -- So that an update from multiple forward/backward training operations is 
  -- simply the sum of the updates from individual forward/backward updates
  -- All the work for the "opfunc" has already been done above, hence its skeletal form.
  -- I prefer this approach to the baroque opfuncs you find in many implementations
  local function opfunc(dummyParams)
    return totalLoss, gradParams
  end
  
  if(debug) then
    parallelSyncAll("Just before model update")  
  end
  
  -- Dereference function pointer to the update method
  updateConfig.method(opfunc, params, updateConfig, updateMethodState)

  -- Note time taken for update
  local timeForUpdate = sys.toc()
  parallelSyncAll("Just after model update")  

  -- Recompute params norm after update
  if(debug or parallel.id == 0) then
    parallel.print("    Params norm = " .. Network.getParamsNorm())
    io.flush()
  end
end

-- Validation
local numValidationSamples = 10
function Network.validateModel(model)
  local error
  local sumError = 0
  local predictions, testingSamples
  
  for sample = 1, numValidationSamples do  
    cutorch.setDevice(model.gpu)
    testingSamples = DataSets.getTestingSamples()
    model.inputBuffer:resize(testingSamples.inputs:size()):copy(testingSamples.inputs)
    predictions = model:forward(model.inputBuffer) 
    error = Evaluate.errorRate(predictions, testingSamples.targets:typeAs(predictions))
    sumError = sumError + error
    
    -- Memory recycling
    predictions = nil
    testingSamples = nil
  end
  
  return sumError / numValidationSamples
end

return Network
