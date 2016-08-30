--------------------------------------------------------------------------------
-- Turn the network's prediction into either even (0) or odd (1) by
-- rounding the prediction values up or down. Then count how many are correct.
--------------------------------------------------------------------------------

require 'torch'

local Evaluate = {}

Evaluate.errorRate = function (predictions, targets)
  local error = 0
  local totalError = 0
  for i = 1, predictions:size(1) do
    error =  targets[i][1]==0 and predictions[i][1]< 0.5 and 0.0 or 
             targets[i][1]==1 and predictions[i][1]>=0.5 and 0.0 or
             1.0
    totalError = totalError + error
  end
  return totalError / predictions:size(1)
end

return Evaluate
