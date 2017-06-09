-----------------------------------------------------------------------
-- Define CNN model
-----------------------------------------------------------------------

require 'nn'
require 'torch'
require 'image'
require 'cunn'

print '==> define model'
numOutputs = 2;
numKernels1 = 256
numKernels2 = 256
filtDim1 = 24
filtDim2 = 5
poolDim =2

-----------------------------------------------------------------------
-- Define model

model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(numChannel, numKernels1, filtDim1, filtDim1,1,1))
model : add(nn.ReLU())
--model : add(nn.Sigmoid())
model : add(nn.SpatialMaxPooling(poolDim,poolDim))

model:add(nn.SpatialConvolutionMM(numKernels1,numKernels2, filtDim2, filtDim2))
model : add(nn.ReLU())


model : add(nn.SpatialMaxPooling(poolDim,poolDim))

model:add( nn.Reshape( 256*4 ) )

model :add(nn.Linear(256*4 ,numOutputs))
model:add(nn.LogSoftMax())
-----------------------------------------------------------------------

print '==> here is the model:'
print(model)

