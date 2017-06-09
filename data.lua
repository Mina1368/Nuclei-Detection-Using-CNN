-----------------------------------------------------------------------
-- Loading Data with .mat format
-- normalizing the Data
-----------------------------------------------------------------------

matio = require 'matio'
require 'torch'
require 'torchx'  -- to use torch.find

-----------------------------------------------------------------------

-- Load the data
print '==> Loading the data'
Data1 = matio.load('data_E_41.mat')

imageDim = 41 --size of each image patch
numChannel = 1 --use grayscale images
numSample = 100100 --number of samples
trsize = .95*numSample --size of train data
tesize = numSample-trsize--size of test data

local train_x = torch.Tensor(imageDim,imageDim,numSample)
local train_y = torch.Tensor(numSample,2)


train_x[{{},{},{1,numSample}}] = Data1.data_E


train_y[{{1,numSample},{}}] = Data1.label

-----------------------------------------------------------------------

print '==> Reshape the data'
-- reshape the data 


-----------------------------------------------------------------------

-- Reshape train data
trainData = torch.Tensor(trsize,numChannel,imageDim,imageDim)
trainLabels = torch.Tensor(trsize)
for i = 1,trsize do
trainData[{i,{},{},{}}] = train_x[{{},{},i}]
index  = torch.find(train_y[{i,{}}],1)
trainLabels[{i}] = index[1]
--trainLabels[{i}] = Data.train_y[{i,{}}]
end

-- Reshape test data
testData = torch.Tensor(tesize,numChannel,imageDim,imageDim)
testLabels = torch.Tensor(tesize)
for i = trsize+1,trsize+tesize do
testData[{i-trsize,{},{},{}}] = train_x[{{},{},i}]
index  = torch.find(train_y[{i,{}}],1)
testLabels[{i-trsize}] = index[1]
end

-----------------------------------------------------------------------

print '==> normalizing the data'
-- nomalizing the data
for i = 1,trsize do
min = trainData[{i,{},{},{}}]:min()
max = trainData[{i,{},{},{}}]:max()
trainData[{i,{},{},{}}] = trainData[{i,{},{},{}}]:add(-min)
trainData[{i,{},{},{}}] = trainData[{i,{},{},{}}]:div(max-min)
end

for i = 1,tesize do
min = testData[{i,{},{},{}}]:min()
max = testData[{i,{},{},{}}]:max()
testData[{i,{},{},{}}] = testData[{i,{},{},{}}]:add(-min)
testData[{i,{},{},{}}] = testData[{i,{},{},{}}]:div(max-min)
end

-----------------------------------------------------------------------


