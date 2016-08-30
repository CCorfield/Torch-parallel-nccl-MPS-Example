--------------------------------------------------------------------------------
-- This module follows the conventions in the "optim" library for converging a 
-- loss function, hence the choice of arguments "opfunc", etc.
-- The implementation of Nesterov acceleration in Torch's optim library doesn't 
-- seem optimal to me. Nesterov noted in 1983, that if a function satisfies 
-- |Df(x) - Df(y)| < beta * |x - y|, then the algorithm implemented below will 
-- satisfy: 
--
--     |f(y_s) - f(x_min)| <= 2*beta*|x_1 - x_min|^2 / s^2
--
-- Where 's' is used for the s-th iteration, and x_1 is the initial point.
-- In other words, for a large number of updates, the distance from the minimum
-- decreases like the square of the number of updates.
-- The subscripts used below can be read as "sm1" == "s minus 1", "sp1" == "s plus 1"
--------------------------------------------------------------------------------

require 'cutorch'

local function lambdaRecurrence(lambda)
  return 0.5*(1 + math.sqrt(1 + 4*lambda*lambda))
end


function ccNesterov(opfunc, x, config, state)
  
  -- Sequences used in the computation of accelerated Nesterov
  local lambda_sm1 = state.lambda_s or 0 -- (lambda_0 = 0)
  local lambda_s   = lambdaRecurrence(lambda_sm1)
  local lambda_sp1 = lambdaRecurrence(lambda_s)  
  local gamma_s    = (1 - lambda_s)/lambda_sp1
  local neval      = state.neval or 0
  
  -- Initialize cached y_1 = x_1
  if(state.y_s == nil) then
    state.y_s = torch.Tensor():typeAs(x):resizeAs(x):copy(x)
  end
  
  -- Standard call to the opfunc
  local fx, dfdx = opfunc(x)
  
  -- First step, which looks like a normal gradient descent, to calculate y_sp1:
  if(state.y_sp1 == nil) then
    state.y_sp1 = torch.Tensor():typeAs(x):resizeAs(x)
  end
  state.y_sp1:copy(x):add(-config.learningRate, dfdx)
  
  -- Second step goes a little further.
  -- (calculates x_sp1, but uses the existing x for brevity)
  x:copy(state.y_sp1:mul(1 - gamma_s):add(gamma_s, state.y_s))
  
  -- Update state variables
  state.lambda_s = lambda_s
  state.neval = neval + 1
  state.y_s:copy(state.y_sp1)
  
  -- Standard return
  return x, {fx}
end
