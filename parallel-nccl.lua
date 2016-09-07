--[[
  This uses a combination of packages to illustrate multi-process inter-GPU data transfer.
  1) The parallel package handles the fork, exec, and messaging functions
  2) The nccl_ffi package provides access to nVidia's nccl libraries.
  These two packages support a Torch multi-process implementation of:
  - Reduce
  - AllReduce
  - Bcast
  "Ranks" are process id's starting with zero for the parent, and incrementing by one for each child.
  It turns out that parallel.id can be used as the value of the "rank" for each process.
  References to the "root" process mean the process with id == 0, i.e. the parent process
  The tests involve the parent sending a value of "parentSendValue" and the children sending values
  corresponding to their parallel.id [note: Sum(1+2+..+n) = (1/2)*n*(n+1)]. 
]]

-- Parent code:
function parent()

  -- Assorted Torch packages:
  require 'torch'
  require 'cutorch'
  require 'optim' 
  require 'parallel'
  require 'sys'

  -- Lua wrapper for nccl library:
  local ffi = require 'ffi'
  local nccl = {}
  nccl.C = require 'nccl_ffi'
  local ncclResult, ncclErrString

  -- Configuration parameters for this test script:
  local parentGPU = 1 -- Usually set to 1
  local numGPUs = cutorch.getDeviceCount()
  local numWorkersPerGPU = 2
  local numProcesses = numGPUs * numWorkersPerGPU
  local debug = true
  local parentSendValue = 10 -- Value for parent to send
  local sumChildSendValues = (numProcesses-1)*numProcesses/2
  local sumAllSendValues = parentSendValue + sumChildSendValues -- What parent should receive
  -- Misc variables
  local replies

  -- Utility function to synchronize all the processes and print an optional message
  local function parallelSyncAll(msg)
    if(parallel.id == 0) then
      parallel.children:join()
      parallel.print(msg)
      io.flush()
    else
      parallel.yield()
      parallel.print(msg)
      io.flush()
    end
  end
  
  if(debug) then
    parallel.print("Test 0.1: GPU # " .. parentGPU)
    parallel.print("Test 0.2: Parallel ID " .. parallel.id)
  end
  
  -- Create parent sendbuff, recvbuff
  cutorch.setDevice(parentGPU)
  local recvbuff = torch.zeros(5):cuda()
  local sendbuff = torch.zeros(5):cuda()
  
  -- Set up workers, count parent as one worker
  parallel.nfork(numProcesses - 1)
  parallel.children:exec(worker)

  -- Send GPU and PID assignments
  -- Note that we allow the parent to be a worker on GPU #1, so there is one less
  -- forked worker on this GPU
  local assignments = {}
  for gpu = 1, numGPUs do 
    if (gpu == 1) then
      -- Parent counts as one worker
      for worker = 1, numWorkersPerGPU -1  do
        local childId = (gpu - 1)*numWorkersPerGPU + worker
        table.insert(assignments, {gpu = gpu, childID = childId, pid = parallel.children[childId].unixid})
      end
    else
      -- Other GPUs are all workers (note the "-1", because of the parent on GPU #1)
      for worker = 1, numWorkersPerGPU do
        local childId = (gpu - 1)*numWorkersPerGPU + worker - 1
        table.insert(assignments, {gpu = gpu, childID = childId, pid = parallel.children[childId].unixid})
      end
    end
  end
  assert(#assignments == numProcesses-1, "Wrong number of assignments vs. processes")
  
  -- Send assignments
  parallel.children:join()
  parallel.children:send(assignments)

  -- Send nccl uniqueID
  local uniqueId = ffi.new('ncclUniqueId')
  nccl.C.ncclGetUniqueId(uniqueId)
  local x = torch.Tensor(128):int()
  for i = 0,127 do
    x[i+1] =  uniqueId.internal[i]
  end
  local s = ""
  for i = 0, 127 do
    if uniqueId.internal[i] == 0 then break end
    s = s .. " " .. uniqueId.internal[i]
  end
  parallel.print("Test 0.3: UniqueId = " .. s)
  parallel.children:join()
  parallel.children:send(x)
 
  -- Init "rank" and "communicator" for parent process
  local comm = ffi.new('ncclComm_t[?]',1)
  nccl.C.ncclCommInitRank(comm,numProcesses,uniqueId,parallel.id)
  parallel.print("Test 0.4: Rank initialized")
  
  -- Reduce using different send/recv buffers ("out of place")
  cutorch.setDevice(parentGPU) 
  sendbuff:fill(parentSendValue)
  recvbuff:fill(0)
  collectgarbage('stop')
  parallelSyncAll("Test 1.1: Just before Reduce (out of place)")
  ncclResult = nccl.C.ncclReduce(sendbuff:data(),recvbuff:data(),
         sendbuff:size(1), nccl.C.ncclFloat,
         nccl.C.ncclSum, 0, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState()))
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 1.2: Just after Reduce (out of place). Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 1.3: Reduce (out of place), sendbuff[1] = ", sendbuff[1]) 
  parallel.print("Test 1.4: Reduce (out of place), recvbuff[1] = ", recvbuff[1])
  parallel.print("Test 1.5: Reduce (out of place), recvbuff[1] should be = ", sumAllSendValues)
  
  -- Reduce using same send/recv buffer ("in place")
  sendbuff:fill(parentSendValue)
  recvbuff:fill(0)
  collectgarbage('stop')
  parallelSyncAll("Test 2.1: Just before Reduce (in place)")
  ncclResult = nccl.C.ncclReduce(sendbuff:data(),sendbuff:data(),
         sendbuff:size(1), nccl.C.ncclFloat,
         nccl.C.ncclSum, 0, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState()))
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 2.2: Just after Reduce (in place). Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 2.3: Reduce (in place), sendbuff[1] = ", sendbuff[1]) 
  parallel.print("Test 2.4: Reduce (in place), sendbuff[1] should be = ", sumAllSendValues)
  
  -- AllReduce using different send/recv buffers ("out of place")
  collectgarbage('stop')
  sendbuff:fill(parentSendValue)
  recvbuff:zero()
  parallelSyncAll("Test 3.1: Just before AllReduce (out of place)")
  ncclResult = nccl.C.ncclAllReduce(sendbuff:data(), recvbuff:data(), 
         sendbuff:size(1), nccl.C.ncclFloat,
         nccl.C.ncclSum, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState()))
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 3.2: Just after AllReduce (out of place). Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 3.3: AllReduce (out of place), sendbuff[1] = ", sendbuff[1]) 
  parallel.print("Test 3.4: AllReduce (out of place), recvbuff[1] = ", recvbuff[1])
  parallel.print("Test 3.5: AllReduce (out of place), recvbuff[1] should be = ", sumAllSendValues)
  
  -- AllReduce using same send/recv buffer ("in place")
  collectgarbage('stop')
  sendbuff:fill(parentSendValue)
  recvbuff:zero()
  parallelSyncAll("Test 4.1: Just before AllReduce (in place)")
  ncclResult = nccl.C.ncclAllReduce(sendbuff:data(), sendbuff:data(), 
         sendbuff:size(1), nccl.C.ncclFloat,
         nccl.C.ncclSum, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState()))
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 4.2: Just after AllReduce (in place). Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 4.3: AllReduce (in place), sendbuff[1] = ", sendbuff[1]) 
  parallel.print("Test 4.4: AllReduce (in place), sendbuff[1] should be = ", sumAllSendValues)
  
  -- Broadcast
  sendbuff:fill(parentSendValue)
  recvbuff:fill(0)
  collectgarbage('stop')
  parallelSyncAll("Test 5.1: Just before Broadcast")
  ncclResult = nccl.C.ncclBcast(sendbuff:data(),  
         sendbuff:size(1), nccl.C.ncclFloat,
         parallel.id, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState()))
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 5.2: Just after Broadcast. Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 5.3: Broadcast, sendbuff[1] = ", sendbuff[1]) 

  -- Sync/terminate when all workers are done
  parallel.children:join('break')
  parallel.print('Test 6.1: All processes terminated')
end

-- Worker code:
function worker()
  -- A worker starts with a blank environment, which means all the packages
  -- must be explicitly loaded

  -- Assorted Torch packages
  require 'torch'
  require 'cutorch'
  require 'sys'
  require 'parallel'

  -- Lua wrapper for nccl library
  local ffi = require 'ffi'
  local nccl = {}
  nccl.C = require 'nccl_ffi'
  local ncclResult, ncclErrString
 
  -- Configuration variables (see parent)
  local myGPU, myPID
  local model, sendbuff, recvbuff
  local copyBuffer
  local debug = true
  local numGPUs = cutorch.getDeviceCount()
  local numProcesses
  local parentSendValue = 10 -- See parent
  local childSendValue = parallel.id
  local sumChildSendValues -- See below
  
  -- Utility function to synch all processes and print optional message
  local function parallelSyncAll(msg)
    if(parallel.id == 0) then
      parallel.children:join()
      parallel.print(msg)
      io.flush()
    else
      parallel.yield()
      parallel.print(msg)
      io.flush()
    end
  end

  -- Receive GPU and PID assignments
  parallel.yield()
  local assignments = parallel.parent:receive()
  numProcesses = #assignments + 1 -- The parent process does not include itself in the assignments
  for i=1,#assignments do
    local gpu = assignments[i].gpu
    local childID = assignments[i].childID
    local pid = assignments[i].pid
    if parallel.id == childID then
      myGPU = gpu
      myPID = pid
      if(debug) then
        parallel.print("Test 0.1: GPU #" .. myGPU)
        parallel.print("Test 0.2: Unix pid " .. myPID)
        parallel.print("Test 0.3: Parallel ID " .. parallel.id)
        io.flush()
      end
    end
  end
  sumChildSendValues = (numProcesses-1)*numProcesses/2
  
  -- Allocate tensors
  cutorch.setDevice(myGPU)
  sendbuff = torch.zeros(5):cuda()
  recvbuff = torch.zeros(5):cuda()
  parallel.print("Test 0.4: Allocated sendbuff, recvbuff")
  io.flush()
  
  -- Receive uniqueId for communicator clique
  local uniqueId = ffi.new('ncclUniqueId')
  parallel.yield()
  local x = parallel.parent:receive()
  local s = ""
  for i = 0,127 do
    uniqueId.internal[i] = x[i+1]
    if x[i+1] ~= 0 then s = s .. " " .. x[i+1] end
  end
  parallel.print("Test 0.5: UniqueId = " .. s)
  io.flush()
  
  -- Init this worker's "rank" and "communicator"
  cutorch.setDevice(myGPU)
  local comm = ffi.new('ncclComm_t[?]',1)
  nccl.C.ncclCommInitRank(comm,numProcesses,uniqueId,parallel.id)
  parallel.print("Test 0.6: Rank initialized")
  io.flush()

  -- Reduce, root using different send/recv buffers ("out of place"):
  -- Note: child uses nil for its recvbuffer
  cutorch.setDevice(myGPU)
  sendbuff:fill(childSendValue)
  recvbuff:fill(0)
  collectgarbage('stop')
  parallelSyncAll("Test 1.1: Just before Reduce (out of place)")
  ncclResult = nccl.C.ncclReduce(sendbuff:data(),nil,
         sendbuff:size(1), nccl.C.ncclFloat,
         nccl.C.ncclSum, 0, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState()))
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 1.2: Just after Reduce (out of place). Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 1.3: Reduce (out of place), sendbuff[1] = ", sendbuff[1])
  io.flush()

  -- Reduce, root using same send/recv buffer ("in place"):
  -- Note: child uses nil for its recvbuffer
  sendbuff:fill(childSendValue)
  recvbuff:fill(0)
  collectgarbage('stop')
  parallelSyncAll("Test 2.1: Just before Reduce (in place)")
  ncclResult = nccl.C.ncclReduce(sendbuff:data(),nil,
         sendbuff:size(1), nccl.C.ncclFloat,
         nccl.C.ncclSum, 0, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState()))
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 2.2: Just after Reduce (in place). Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 2.3: Reduce (in place), sendbuff[1] = ", sendbuff[1])
  io.flush()
    
  -- AllReduce using different send/recv buffers ("out of place"):
  sendbuff:fill(childSendValue)
  recvbuff:fill(0)
  collectgarbage('stop')
  parallelSyncAll("Test 3.1: Just before AllReduce (out of place)")
  ncclResult = nccl.C.ncclAllReduce(sendbuff:data(), recvbuff:data(), 
         sendbuff:size(1), nccl.C.ncclFloat,
         nccl.C.ncclSum, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState())) 
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 3.2: Just after AllReduce (out of place). Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 3.3: AllReduce (out of place), sendbuff = ", sendbuff[1])
  parallel.print("Test 3.4: AllReduce (out of place), recvbuff = ", recvbuff[1])
  parallel.print("Test 3.5: AllReduce (out of place), recvbuff should be = ", parentSendValue + sumChildSendValues)
  io.flush()
  
  -- AllReduce using same send/recv buffer ("in place"):
  sendbuff:fill(childSendValue) -- reset value
  recvbuff:fill(0)
  collectgarbage('stop')
  parallelSyncAll("Test 4.1: Just before AllReduce (in place)")
  ncclResult = nccl.C.ncclAllReduce(sendbuff:data(), sendbuff:data(), 
         sendbuff:size(1), nccl.C.ncclFloat,
         nccl.C.ncclSum, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState())) 
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 4.2: Just after AllReduce (in place). Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 4.3: AllReduce (in place), send & recv buff[1] = ", sendbuff[1])
  parallel.print("Test 4.4: AllReduce (in place), send & recv buff[1] should be = ", parentSendValue + sumChildSendValues)
  io.flush()
  
  -- Broadcast:
  sendbuff:fill(parallel.id)
  recvbuff:fill(0)
  collectgarbage('stop')
  parallelSyncAll("Test 5.1: Just before Broadcast")
  ncclResult = nccl.C.ncclBcast(recvbuff:data(), 
         recvbuff:size(1), nccl.C.ncclFloat,
         0, comm[0], 
         ffi.C.THCState_getCurrentStream(cutorch.getState())) 
  ncclErrString = ffi.string(nccl.C.ncclGetErrorString(ncclResult))
  parallelSyncAll("Test 5.2: Just after Broadcast. Result = " .. ncclErrString)
  collectgarbage('restart')
  parallel.print("Test 5.3: Broadcast, recvbuff[1] = ", recvbuff[1])
  parallel.print("Test 5.4: Broadcast, recvbuff[1] should be = ", parentSendValue)
  io.flush()
  
  -- Receive terminations message
  local msg = parallel.yield()
  parallel.print("Test 6.1: Received " .. msg .. ". Exiting")
  io.flush()
  
  collectgarbage()
end

-- Protected execution:
local ok,err = pcall(parent)
if not ok then print(err) end
parallel.close()
