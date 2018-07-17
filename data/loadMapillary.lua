----------------------------------------------------------------------
-- Mapillary data loader,
-- Abhishek Chaurasia,
-- August 2016
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------

local trsize, tesize

-- TODO: only for the sample, has to change it
trsize = 18000
tesize = 2000

-- TODO: change this to mapillary annotations
-- classes are only one more class than conClasses, the "Unlabeled"
-- More complete definition is:
--      Classes that does not fit, or appear infrequently
--      vehicles, pedestrains
--      Road: tunnel
--      Shoulder, or cross walks
--      Poles, vegitation that is not the boundary of the road, not critical to driving, thus excluding fences
--      Sky, mountain
--      Lane Markings
--                 1          2         3            4                5               6      7
local classes = {'Ignored', 'Movable', 'Navigable', 'NoneNavigable', 'StaticLayout', 'Sky', 'Lane'}
local conClasses = {'Movable', 'Navigable', 'NoneNavigable', 'StaticLayout', 'Sky', 'Lane'}

local nClasses = #classes

--------------------------------------------------------------------------------
-- Ignoring unnecessary classes
-- Map the original label to the Classes label space, with 1 being ignored.

local classMap = {[-1] =  {1}, -- animal--bird
                  [0]  =  {2}, -- "animal--ground-animal"
                  [1]  =  {4}, -- "construction--barrier--curb"
                  [2]  =  {4}, -- "construction--barrier--fence"
                  [3]  =  {4}, -- "construction--barrier--guard-rail"
                  [4]  =  {4}, -- "construction--barrier--other-barrier"
                  [5]  =  {4}, -- "construction--barrier--wall"
                  [6]  =  {4}, -- "construction--flat--bike-lane"
                  [7]  =  {4}, -- "construction--flat--crosswalk-plain"
                  [8]  =  {4}, -- "construction--flat--curb-cut"
                  [9]  =  {4}, -- "construction--flat--parking"
                  [10] =  {4}, -- "construction--flat--pedestrian-area"
                  [11] =  {4}, -- "construction--flat--rail-track"
                  [12] =  {3}, -- "construction--flat--road"
                  [13] =  {3}, -- "construction--flat--service-lane"
                  [14] =  {4}, -- "construction--flat--sidewalk"
                  [15] =  {5}, -- "construction--structure--bridge"
                  [16] =  {5}, -- "construction--structure--building"
                  [17] =  {3}, -- "construction--structure--tunnel"
                  [18] =  {2}, -- "human--person"
                  [19] =  {2}, -- "human--rider--bicyclist"
                  [20] =  {2}, -- "human--rider--motorcyclist"
                  [21] = {2}, -- "human--rider--other-rider"
                  [22] = {7}, -- "marking--crosswalk-zebra"
                  [23] = {7}, -- "marking--general"
                  [24] = {6}, -- "nature--mountain"
                  [25] = {1}, -- "nature--sand" Ignored, due to rare to see
                  [26] = {6}, -- "nature--sky"
                  [27] = {1}, -- "nature--snow" Not sure whether snow mountain or snow on road
                  [28] = {1}, -- "nature--terrain" Ignored due to rare appearance
                  [29] =  {5}, -- "nature--vegetation"
                  [30] =  {4}, -- "nature--water"
                  [31] = {5}, -- "object--banner"
                  [32] = {5}, -- "object--bench"
                  [33] = {5}, -- "object--bike-rack"
                  [34]  =  {5}, -- "object--billboard"
                  [35]  =  {1}, -- "object--catch-basin"  Ignored since not frequent
                  [36]  =  {1}, -- "object--cctv-camera"  Ignored since not frequent
                  [37]  =  {5}, -- "object--fire-hydrant"
                  [38]  =  {5}, -- "object--junction-box"
                  [39]  =  {5}, -- "object--mailbox"
                  [40]  =  {3}, -- "object--manhole"
                  [41]  =  {5}, -- "object--phone-booth"
                  [42]  =  {1}, -- "object--pothole" Ignored, since not frequent
                  [43] =  {5}, -- "object--street-light"
                  [44] =  {5}, -- "object--support--pole"
                  [45] =  {5}, -- "object--support--traffic-sign-frame"
                  [46] =  {5}, -- "object--support--utility-pole"
                  [47] =  {5}, -- "object--traffic-light"
                  [48] =  {5}, -- "object--traffic-sign--back"
                  [49] =  {5}, -- "object--traffic-sign--front"
                  [50] =  {5}, -- "object--trash-can"
                  [51] =  {2}, -- "object--vehicle--bicycle"
                  [52] =  {1}, -- "object--vehicle--boat" Ignoring boat
                  [53] =  {2}, -- "object--vehicle--bus"
                  [54] = {2}, -- "object--vehicle--car"
                  [55] = {2}, -- "object--vehicle--caravan"
                  [56] = {2}, -- "object--vehicle--motorcycle"
                  [57] = {2}, -- "object--vehicle--on-rails"
                  [58] = {2}, -- "object--vehicle--other-vehicle"
                  [59] = {2}, -- "object--vehicle--trailer"
                  [60] = {2}, -- "object--vehicle--truck"
                  [61] = {2}, -- "object--vehicle--wheeled-slow"
                  [62] =  {1}, -- "void--car-mount"
                  [63] =  {1}, -- "void--ego-vehicle"
                  [64] = {1}, -- "void--unlabeled"
                              }

-- saving training histogram of classes
local histClasses = torch.Tensor(#classes):zero()

--------------------------------------------------------------------------------
print '\n\27[31m\27[4mLoading mapillary dataset\27[0m'
print('# of classes: ' .. #classes)

local trainData, testData
local loadedFromCache = false
local dirName = opt.imHeight .. '_' .. opt.imWidth
paths.mkdir(paths.concat(opt.cachepath, dirName))
local cityscapeCachePath = paths.concat(opt.cachepath, dirName, 'data.t7')

if opt.cachepath ~= "none" and paths.filep(cityscapeCachePath) then
   print('\27[32mData cache found at: \27[0m\27[4m' .. cityscapeCachePath .. '\27[0m')
   local dataCache = torch.load(cityscapeCachePath)
   trainData = dataCache.trainData
   testData = dataCache.testData
   histClasses = dataCache.histClasses
   loadedFromCache = true
   dataCache = nil
   collectgarbage()
else
   print("Classes are:")
   print(classes)
   local function has_image_extensions(filename)
      local ext = string.lower(path.extension(filename))

      -- compare with list of image extensions
      local img_extensions = {'.jpeg', '.jpg', '.png', '.ppm', '.pgm'}
      for i = 1, #img_extensions do
         if ext == img_extensions[i] then
            return true
         end
      end
      return false
   end

   -- initialize data structures:
   trainData = {
      data = torch.FloatTensor(trsize, opt.channels, opt.imHeight, opt.imWidth),
      labels = torch.FloatTensor(trsize, opt.imHeight, opt.imWidth),
      preverror = 1e10, -- a really huge number
      size = function() return trsize end
   }

   testData = {
      data = torch.FloatTensor(tesize, opt.channels, opt.imHeight, opt.imWidth),
      labels = torch.FloatTensor(tesize, opt.imHeight, opt.imWidth),
      preverror = 1e10, -- a really huge number
      size = function() return tesize end
   }

   print('==> Loading training files')

   local dpathRoot = opt.datapath .. '/training/'

   assert(paths.dirp(dpathRoot), 'No training folder found at: ' .. opt.datapath)
   --load training images and labels:
   local c = 1
    local dir = "/images/"
  local dpath = dpathRoot .. dir .. '/'
  for file in paths.iterfiles(dpath) do

     -- process each image
     if has_image_extensions(file) and c <= trsize then
        local imgPath = path.join(dpath, file)

        --load training images:
        local dataTemp = image.load(imgPath)
        trainData.data[c] = image.scale(dataTemp,opt.imWidth, opt.imHeight)

        -- Load training labels:
        -- Load labels with same filename as input image.
        imgPath = string.gsub(imgPath, "images", "labels_converted")
        imgPath = imgPath:sub(1, -5) .. ".png"

        -- label image data are resized to be [1,nClasses] in [0 255] scale:
        local labelIn = image.load(imgPath, 1, 'byte')
        local labelFile = image.scale(labelIn, opt.imWidth, opt.imHeight, 'simple'):float()

        labelFile:apply(function(x) return classMap[x-1][1] end)

        -- Syntax: histc(data, bins, min, max)
        histClasses = histClasses + torch.histc(labelFile, #classes, 1, #classes)

        -- convert to int and write to data structure:
        trainData.labels[c] = labelFile

        c = c + 1
        if c % 20 == 0 then
           xlua.progress(c, trsize)
        end
        collectgarbage()
     end
  end

   print('')

   print('==> Loading testing files')
   dpathRoot = opt.datapath .. '/validation/'

   assert(paths.dirp(dpathRoot), 'No testing folder found at: ' .. opt.datapath)
   -- load test images and labels:
   local c = 1
   local dpath = dpathRoot  .. '/images/'

  for file in paths.iterfiles(dpath) do

     -- process each image
     if has_image_extensions(file) and c <= tesize then
        local imgPath = path.join(dpath, file)

        --load training images:
        local dataTemp = image.load(imgPath)
        testData.data[c] = image.scale(dataTemp, opt.imWidth, opt.imHeight)

        -- Load validation labels:
        -- Load labels with same filename as input image.
        imgPath = string.gsub(imgPath, "images", "labels_converted")
        imgPath = imgPath:sub(1, -5) .. ".png"


        -- load test labels:
        -- label image data are resized to be [1,nClasses] in in [0 255] scale:
        local labelIn = image.load(imgPath, 1, 'byte')
        local labelFile = image.scale(labelIn, opt.imWidth, opt.imHeight, 'simple'):float()


        labelFile:apply(function(x) return classMap[x-1][1] end)

        -- convert to int and write to data structure:
        testData.labels[c] = labelFile

        c = c + 1
        if c % 20 == 0 then
           xlua.progress(c, tesize)
        end
        collectgarbage()
     end
  end

end

if opt.cachepath ~= "none" and not loadedFromCache then
   print('\27[32m'..'==> Saving data to cache: \27[0m' .. cityscapeCachePath)
   local dataCache = {
      trainData = trainData,
      testData = testData,
      histClasses = histClasses
   }
   torch.save(cityscapeCachePath, dataCache)
   dataCache = nil
   collectgarbage()
end

----------------------------------------------------------------------
print '==> Normalizing data'

-- It's always good practice to verify that data is properly
-- normalized.
local trainMean = torch.zeros(3, trainData.data:size(3), trainData.data:size(4))
for i = 1, opt.channels do
   trainMean[i] = trainData.data[{{}, i, {}, {}}]:mean()
end

for i = 1, trainData.data:size(1) do
   trainData.data[i]:add(-trainMean)
end
for i = 1, testData.data:size(1) do
   testData.data[i]:add(-trainMean)
end
torch.save(paths.concat(opt.cachepath, dirName, 'stat.t7'), trainMean)

----------------------------------------------------------------------

local classes_td = {[1] = 'classes,targets\n'}
for _,cat in pairs(classes) do
   table.insert(classes_td, cat .. ',1\n')
end

local file = io.open(paths.concat(opt.save, 'categories.txt'), 'w')
file:write(table.concat(classes_td))
file:close()

-- Exports
opt.dataClasses = classes
opt.dataconClasses  = conClasses
opt.datahistClasses = histClasses

return {
   trainData = trainData,
   testData = testData,
}
