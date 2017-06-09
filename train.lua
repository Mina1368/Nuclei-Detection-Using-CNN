-----------------------------------------------------------------------
-- Training procedure
-- Here we use SGD optimization method for training
-----------------------------------------------------------------------
require 'nn'
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

-----------------------------------------------------------------------

print '==> configuring optimizer'
learningRate = 1e-4
weightDecay = 0
momentum = 0
learningRateDecay = 1e-7
coefl1 = .01
coefl2 = 1
   model:cuda()
   criterion:cuda()
if model then
   parameters,gradParameters = model:getParameters()
end

-----------------------------------------------------------------------

print '==> defining training procedure'

-- classes
classes = {'1','2'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat('result', 'train.log'))
testLogger = optim.Logger(paths.concat('result', 'test.log'))

function train()
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')

   for t = 1,trsize,128 do
      -- disp progress
      xlua.progress(t, trsize)

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+256-1,trsize) do
         -- load new sample
         local input = trainData[shuffle[i]]
         local target = trainLabels[shuffle[i]]
input = input:cuda() 
         table.insert(inputs, input)
         table.insert(targets, target)
      end
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err


                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)
                          -- update confusion
                          confusion:add(output, targets[i])

                       end
 local norm,sign= torch.norm,torch.sign
-- regu penalty l1 and l2
f = f + coefl1*norm(parameters,1)
f = f + coefl2*norm(parameters,2)^2/2

 gradParameters:add( sign(parameters):mul(coefl1) + parameters:clone():mul(coefl2) )

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch

         optim.sgd(feval, parameters, optimState)

   end

   -- time taken
   time = sys.clock() - time
   time = time / trsize
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   -- save/log current net

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

   local filename = paths.concat('result', 'model_41.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   confusion:zero()
   epoch = epoch + 1
end
