-- a program to: 
-- 		load the model once
--		read a image from disk, perform pre-processing, run the network on the gpu, grab the output
-- 		save the prediction to disk

require 'pl'
require 'nn'

----------------------------------------------------------------------
-- Local repo files
local opts = require 'opts'
opt = opts.parse(arg)

-- nb of threads and fixed seed (for repeatable experiments)
-- torch.setnumthreads(opt.threads)
torch.manualSeed(12)
torch.setdefaulttensortype('torch.FloatTensor')

-- print('==> switching to CUDA')
require 'cudnn'
require 'cunn'
cutorch.setDevice(opt.devid)

----------------------------------------------------------------------
local data, chunks, ft


local classes = {'Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
                 'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
                 'Sky', 'Person', 'Rider', 'Car', 'Truck',
                 'Bus', 'Train', 'Motorcycle', 'Bicycle'}
local conClasses = {'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
                    'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
                    'Sky', 'Person', 'Rider','Car', 'Truck',
                    'Bus', 'Train', 'Motorcycle', 'Bicycle'} -- 19 classes
opt.dataClasses = classes
opt.dataconClasses  = conClasses
opt.datahistClasses = torch.Tensor(#classes):zero()
local nClasses = #classes

require 'image'
-- TODO: fake the data structure, by looking at data/loadCityScapes
local path = "/home/gaoyang1/data/CityScapes/leftImg8bit/val/munster/munster_000115_000019_leftImg8bit.png"
local one = image.load(path)
one = image.scale(one, opt.imWidth, opt.imHeight)
local trainMeanPath = "/home/gaoyang1/data/CityScapes/cache/512_1024/stat.t7"
local trainMean = torch.load(trainMeanPath)

one:add(-trainMean)

----------------------------------------------------------------------

t = paths.dofile(opt.model)

-- TODO: set batch size to 1
print('[batchSize = ' ..  opt.batchSize .. ']')

local loss = t.loss
local model=torch.load("/home/gaoyang1/linknet/Final/model-cs-IoU.net")
trainError=0.0
model:evaluate()

assert(opt.batchSize == 1)

local x = torch.Tensor(opt.batchSize, opt.channels, opt.imHeight, opt.imWidth)
x = x:cuda()
x[1] = one

local y = model:forward(x)
-- y has a shape of 1*#class*64*64 size. 

local y = y:transpose(2, 4):transpose(2, 3)
-- now has size 1*64*64*#class
y = y:reshape(y:numel()/y:size(4), #classes):sub(1, -1, 2, #opt.dataClasses)
local _, predictions = y:max(2)
predictions = predictions:view(-1)
predictions = predictions:view(1, opt.imHeight, opt.imWidth)
predictions =predictions:type('torch.FloatTensor')
predictions = predictions:mul(1.0/19) 
image.save("one_image_pred.png", predictions)
torch.save("one_image_test_output.t7", predictions)

