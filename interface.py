import lutorpy as lua
import numpy as np
from PIL import Image
from collections import defaultdict
import time, os

class Segmenter:
	def init(self):
		lua.LuaRuntime(zero_based_index=False)
		lua.execute("arg = {}")
		self.segment_func=require('interface')
	
	def segment(self, image):
		# assume image has size H*W*3
		image = np.transpose(image, (2, 0, 1))
		image = image.astype(np.float32)
		image = image / 255.0

		image = torch.fromNumpyArray(image)
		out = self.segment_func(image)
		out = out.asNumpyArray()

		# minus 1 for converting 1-19 to 0-18 classes
		return out-1

	def colorize(self, pred):
	    color = {0:[128, 64, 128], 1:[244, 35,232], 2:[ 70, 70, 70],
	             3:[102, 102,156], 4:[190,153,153], 5:[153,153,153],
	             6:[250, 170, 30], 7:[220,220,  0], 8:[107,142, 35],
	             9:[152,251, 152], 10:[70,130,180], 11:[220, 20,60],
	             12:[255,  0,  0], 13:[0, 0,  142], 14:[0,  0,  70],
	             15:[0, 60,  100], 16:[0, 80, 100], 17:[0,  0, 230],
	             18:[119, 11, 32]
	             }
	    color = defaultdict(lambda: [0,0,0], color)
	    shape = pred.shape
	    pred = pred.ravel()
	    pred = np.asarray([color[i] for i in pred])
	    pred = pred.reshape(shape[0],shape[1],3)

	    return pred.astype(np.uint8)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# TODO: test a image from nexar camera
#path = "/home/gaoyang1/data/CityScapes/leftImg8bit/val/munster/munster_000115_000019_leftImg8bit.png"
#path = "/scratch/yang/aws_data/bdd100k/yolo_format/images/val/c8620a67-55f86ae2.jpg"
path = "/scratch/yang/aws_data/mapillary/validation/images/OYyFv3XcyrSla0sgF6JrEg.jpg"

img = Image.open(path)
#print("raw size", img.shape)
#img = img.resize((1024, 512))
img = img.resize((768, 576))
img = np.array(img)

print("resized:", img.shape)
seg=Segmenter()
seg.init()

start = time.time()
for i in range(100):
	pred = seg.segment(img)
	print(i)
print("elapsed time:", time.time()-start)

colored = seg.colorize(pred)

colored = Image.fromarray(colored)
colored.save("ss_out.png")