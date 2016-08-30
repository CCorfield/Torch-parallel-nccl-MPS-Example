--------------------------------------------------------------------------------
-- Load a saved model and run test samples through it.
--------------------------------------------------------------------------------
require 'torch'
require 'cutorch'
local DataSets = require 'DataSets'
local Model = require 'Model'


local loadFile = "ExampleModel.t7" -- Saved by the training scripts
local model = torch.load(loadFile)
model:evaluate()
local type = model.output:type() -- Used below to convert "inputs" to a compatible type

local function predictionString(x)
  return x <= 0.5 and "< 0.5 => even" or "> 0.5 => odd "
end

-- Run through some randomly generated inputs and see what the net predicts:
for i = 1,10 do
  -- Note that "samples" only contains one member
  local samples = DataSets.getTestingSamples()
  local predictions = model:forward(samples.inputs:type(type)):clamp(0,1)
  -- torch.max() has a variety of behaviors; note that "index" is actually a 
  -- (degenerate) tensor, hence the dereference in the print() statement
  local _,index = torch.max(samples.inputs[1],1)
  print (string.format("Input: %d, Prediction: %.2f (%s)", 
         index[1], predictions[1][1],predictionString(predictions[1][1])))
end
