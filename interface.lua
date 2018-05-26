-- a clean up version of the lua interface
require 'pl'
require 'nn'
require 'image'

----------------------------------------------------------------------
-- Local repo files
local opts = require 'opts'
opt = opts.parse(arg)

-- outside variables
-- TODO: have a global config file that points all models
-- TODO: retrain the model on the mappilary data with lane markings
-- TODO: test segmentation with other image size: different size doesn't work
opt.devid = 1
local trainMeanPath = "/home/gaoyang1/data/CityScapes/cache/512_1024/stat.t7"
local pretrainedModel = "/home/gaoyang1/linknet/Final/model-cs-IoU.net"
opt.imHeight = 512
opt.imWidth =  1024
-- outside variables end
opt.batchSize = 1
opt.nGPU = 1

-- nb of threads and fixed seed (for repeatable experiments)
-- torch.setnumthreads(opt.threads)
torch.manualSeed(12)
torch.setdefaulttensortype('torch.FloatTensor')

-- print('==> switching to CUDA')
require 'cudnn'
require 'cunn'
cutorch.setDevice(opt.devid)

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

-- one time loading job here
local trainMean = torch.load(trainMeanPath)
-- from image wide mean to pixel wide mean
trainMean = trainMean:mean(2):mean(3)
local model=torch.load(pretrainedModel)
model:evaluate()
local x = torch.Tensor(opt.batchSize, opt.channels, opt.imHeight, opt.imWidth)
x = x:cuda()


-- interactive function here
function segment(image)
	-- the image could be any size, a float array (0-1) with size 3*H*W
	-- convert to float type, transpose
	image:add(-trainMean:expandAs(image))
	x[1] = image
	local y = model:forward(x)
	-- y has a shape of 1*#class*H*W size. 

	local y = y:transpose(2, 4):transpose(2, 3)
	-- now has size 1*H*W*#class
	y = y:reshape(y:numel()/y:size(4), #classes):sub(1, -1, 2, #opt.dataClasses)
	local _, predictions = y:max(2)
	predictions = predictions:view(opt.imHeight, opt.imWidth)
	predictions =predictions:type('torch.IntTensor')
	return predictions
end

return segment