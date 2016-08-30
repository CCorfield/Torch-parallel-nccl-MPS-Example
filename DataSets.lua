--------------------------------------------------------------------------------
-- Generate training and testing samples for the network.
-- We follow the "mini batch" approach, so that the first dimension is used to
-- index the batch, and the second dimension is used for the data.
-- In the reference example, the mini batches have only one member, but real 
-- world applications will have more.
--------------------------------------------------------------------------------

require 'torch'
local debug = false

local DataSets = {}

local dimi = 10

-- Public Function
local testingSamples = {}
testingSamples.inputs = torch.Tensor(1,dimi)
testingSamples.targets = torch.Tensor(1,1)

-- Returns a trivial minibatch with one member (but could be more)
function DataSets.getTestingSamples()
  local number = math.random(1,dimi)
  testingSamples.inputs:zero()
  testingSamples.inputs[1][number] = 1
  testingSamples.targets[1][1] = number%2
  return testingSamples
end

-- Public Function
local trainingSamples = {}
trainingSamples.inputs = torch.Tensor(1,dimi)
trainingSamples.targets = torch.Tensor(1,1)

-- Returns a trivial minibatch with one member (but could be more)
function DataSets.getTrainingSamples()
  local number = math.random(1,dimi)
  trainingSamples.inputs:zero()
  trainingSamples.inputs[1][number] = 1
  trainingSamples.targets[1][1] = number%2
  return trainingSamples
end

local numberOfTrainingSamples = 10 -- Completely arbitrary number for this example
function DataSets.getNumberOfTrainingSamples()
  return numberOfTrainingSamples
end

return DataSets
