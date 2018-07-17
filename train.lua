----------------------------------------------------------------------
-- This script
--   + constructs mini-batches on the fly
--   + computes model error
--   + optimizes the error using several optmization
--     methods: SGD, L-BFGS, ADAM.
--
-- Written by  : Abhishek Chaurasia, Eugenio Culurciello
-- Dated       : August, 2016
----------------------------------------------------------------------

require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

torch.setdefaulttensortype('torch.FloatTensor')

local loss = t.loss

----------------------------------------------------------------------

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local model = t.model
print '==> Flattening model parameters'
local w,dE_dw = model:getParameters()

----------------------------------------------------------------------
local confusion
if opt.dataconClasses then
   print('\27[31mClass \'Unlabeled\' is ignored in confusion matrix\27[0m')
   confusion = optim.ConfusionMatrix(opt.dataconClasses)
else
   confusion = optim.ConfusionMatrix(opt.dataClasses)
end

local learningRateSteps = {0.5e-4, 0.1e-4, 0.5e-5, 0.1e-6}
local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   learningRateDecay = opt.learningRateDecay
}

----------------------------------------------------------------------
print '==> Allocating minibatch memory'
local x = torch.Tensor(opt.batchSize, opt.channels, opt.imHeight, opt.imWidth)
local yt = torch.Tensor(opt.batchSize, opt.imHeight, opt.imWidth)

x = x:cuda()
yt = yt:cuda()

local function train(trainData, classes, epoch)
   if epoch % opt.lrDecayEvery == 0 then
      optimState.learningRate = optimState.learningRate * opt.learningRateDecay
   end

   local time = sys.clock()

   -- total loss error
   local err
   local totalerr = 0

   -- This matrix records the current confusion across classes
   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size())

   model:training()
   for t = 1, trainData:size(), opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- batch fits?
      if (t + opt.batchSize - 1) > trainData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x[idx] = trainData.data[shuffle[i]]
         yt[idx] = trainData.labels[shuffle[i]]
         idx = idx + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         model:zeroGradParameters()

         -- evaluate function for complete mini batch
         local y = model:forward(x)
         -- estimate df/dW
         err = loss:forward(y,yt)            -- updateOutput
         local dE_dy = loss:backward(y,yt)   -- updateGradInput
         model:backward(x,dE_dy)
         -- Don't add this to err, so models with different WD
         -- settings can be easily compared. optim functions
         -- care only about the gradient anyway (adam/rmsprop)
         -- dE_dw:add(opt.weightDecay, w)

         -- return f and df/dX
         return err, dE_dw
      end

      -- optimize on current mini-batch
      local _, errt = optim.rmsprop(eval_E, w, optimState)
      -- local _, errt = optim.adam(eval_E, w)

      if opt.saveTrainConf then
         -- update confusion
         model:evaluate()
         local y = model:forward(x):transpose(2, 4):transpose(2, 3)
         y = y:reshape(y:numel()/y:size(4), #classes):sub(1, -1, 2, #opt.dataClasses)
         local _, predictions = y:max(2)
         predictions = predictions:view(-1)
         local k = yt:view(-1)
         if opt.dataconClasses then k = k - 1 end
         confusion:batchAdd(predictions, k)
         model:training()
      end

      totalerr = totalerr + err

   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print(string.format('\n==> Time to learn 1 sample = %2.2f, %s', (time*1000), 'ms'))

   -- print average error in train dataset
   totalerr = totalerr / (trainData:size()*(#opt.dataconClasses)/opt.batchSize)

   trainError = totalerr
   collectgarbage()
   return confusion, model, loss
end

-- Export:
return train
