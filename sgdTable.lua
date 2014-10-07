--[[ A plain implementation of SGD

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.learningRates`      : vector of individual learning rates

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

(Clement Farabet, 2012)
]]
-- this version should still work if the parameters are not flattened
function optim.sgdTable(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   numTensors = #x

   for index=1,numTensors do 
      cutorch.setDevice(cutorch.getTensorDevice(x[index]))
      -- (2) weight decay
      if wd ~= 0 then
         dfdx[index]:add(wd, x[index])
      end

      -- (3) apply momentum
      if mom ~= 0 then
         if not state.dfdx then state.dfdx={} end
         if not state.dfdx[index] then
            state.dfdx[index] = torch.Tensor():typeAs(dfdx[index]):resizeAs(dfdx[index]):copy(dfdx[index])
         else
            state.dfdx[index]:mul(mom):add(1-damp, dfdx[index])
         end
         if nesterov then
            dfdx[index]:add(mom, state.dfdx[index])
         else
            dfdx[index] = state.dfdx[index]
         end
      end

      -- (4) learning rate decay (annealing)
      local clr = lr / (1 + nevals*lrd)
      
      -- (5) parameter update with single or individual learning rates
      if lrs then
         if not state.deltaParameters then state.deltaParameters = {} end
         if not state.deltaParameters[index] then
            state.deltaParameters[index] = torch.Tensor():typeAs(x[index]):resizeAs(dfdx[index])
         end
         state.deltaParameters[index]:copy(lrs):cmul(dfdx[index])
         x[index]:add(-clr, state.deltaParameters[index])
      else
         x[index]:add(-clr, dfdx[index])
      end
   end

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
