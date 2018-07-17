-- a clean up version of the lua interface
require 'pl'
require 'nn'
require 'image'

----------------------------------------------------------------------
-- Local repo files
local opts = require 'opts'
opt = opts.parse(arg)

-- outside variables
opt.devid = 1
-- Those two variables passed from outside
--local trainMeanPath = "/data2/yang_cache/aws_data/linknet/stat.t7"
--local pretrainedModel = "/data2/yang_cache/aws_data/linknet/model-cs-IoU.net"
opt.imHeight = 512
opt.imWidth =  1024
local classes = {'Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
                 'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
                 'Sky', 'Person', 'Rider', 'Car', 'Truck',
                 'Bus', 'Train', 'Motorcycle', 'Bicycle'}
local conClasses = {'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
                    'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
                    'Sky', 'Person', 'Rider','Car', 'Truck',
                    'Bus', 'Train', 'Motorcycle', 'Bicycle'} -- 19 classes

-- Testing the Mapillary Model
-- Those two variables passed from outside
--local trainMeanPath = "/scratch/yang/aws_data/mapillary/cache/576_768/stat.t7"
--local pretrainedModel = "/scratch/yang/aws_data/mapillary/linknet_output2/model-last.net"
opt.imHeight = 576
opt.imWidth =  768
local classes = {'Ignored', 'Movable', 'Navigable', 'NoneNavigable', 'StaticLayout', 'Sky', 'Lane'}
local conClasses = {'Movable', 'Navigable', 'NoneNavigable', 'StaticLayout', 'Sky', 'Lane'}

-- outside variables end
opt.batchSize = batch_size
opt.nGPU = 1

-- nb of threads and fixed seed (for repeatable experiments)
-- torch.setnumthreads(opt.threads)
torch.manualSeed(12)
torch.setdefaulttensortype('torch.FloatTensor')

-- print('==> switching to CUDA')
require 'cudnn'
require 'cunn'
cutorch.setDevice(opt.devid)

opt.dataClasses = classes
opt.dataconClasses  = conClasses
opt.datahistClasses = torch.Tensor(#classes):zero()

-- one time loading job here
local trainMean = torch.load(trainMeanPath)
-- from image wide mean to pixel wide mean
trainMean = trainMean:mean(2):mean(3)
trainMean = trainMean:resize(1,3,1,1)

local model=torch.load(pretrainedModel)
model:evaluate()
local x = torch.Tensor(opt.batchSize, opt.channels, opt.imHeight, opt.imWidth)
x = x:cuda()

trainMean = -trainMean:expandAs(x)
trainMean = trainMean:cuda()

-- interactive function here
function segment(image, output_downsample_factor)
	x = x:copy(image)
	x:add(trainMean)
	local y = model:forward(x)

	-- y has a shape of batch*#class*H*W size.
	-- downsample this output
	local index_h = torch.range(1, opt.imHeight, output_downsample_factor):long()
	local y = y:index(3, index_h)
	local index_w = torch.range(1, opt.imWidth, output_downsample_factor):long()
	local y = y:index(4, index_w)
	-- finish downsampling
	local y = y:transpose(2, 4):transpose(2, 3)

	-- now has size batch*H*W*#class
	y = y:reshape(y:numel()/y:size(4), #classes):sub(1, -1, 2, #opt.dataClasses)
	-- now has shape batchHW*(numclasses-1)
	y = y:contiguous()
	y = y:resize(opt.batchSize, index_h:size()[1], index_w:size()[1], #classes-1)
	-- now has shape batch*H*W*(#classes-1)
	return y
end

return segment