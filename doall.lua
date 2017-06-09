print '==> executing all'
require 'nn'
 require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')


dofile 'data.lua'
dofile 'model.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
print '==> training!'
i=1

while true do
   train()
   test()
end

