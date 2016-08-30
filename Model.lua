--------------------------------------------------------------------------------
-- A simple two layer linear network that we train to spot odd and even integers
-- in the range [1,10]
--------------------------------------------------------------------------------

require 'nn'

local Model = {}

local dimi = 10 -- Size of 1-hot input vector for training with integers in the range [1,10]
local dimo =  1 -- Size of output vector, where 0 => even, and 1 => odd
local dimh = 20 -- Number of hidden units

function Model.init()
  local model = nn.Sequential():add(nn.Linear(dimi, dimh)):add(nn.Linear(20,dimo)) 
  model.criterion = nn.MSECriterion() -- Mean square error
  return model
end

return Model
