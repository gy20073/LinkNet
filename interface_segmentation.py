import lutorpy as lua
import numpy as np
from PIL import Image
import time, os, inspect
from common import resize_images


def get_current_folder():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class Segmenter:
    def __init__(self,
                 model_path, # the path to a pretrained model, use city2eigen
                 mean_path,
                 GPU="0",
                 compute_method="compute_logits",
                 viz_method="visualize_logits",
                 batch_size=1,
                 output_downsample_factor=4):
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU
        self.compute_method = compute_method
        self.viz_method = viz_method
        self.height = 576
        self.width = 768
        self.batch_size = batch_size
        self.output_downsample_factor = output_downsample_factor

        # allow to import the interface.lua
        os.environ["LUA_PATH"] += ";" + get_current_folder() + "/?.lua"
        lua.LuaRuntime(zero_based_index=False)
        lua.execute("arg = {}")
        lua.execute('trainMeanPath="' + mean_path + '"')
        lua.execute('pretrainedModel="' + model_path + '"')
        lua.execute('batch_size=' + str(batch_size))
        self.segment_func = require('interface')

    def compute(self, image):
        return getattr(self, self.compute_method)(image)

    def visualize(self, pred, ibatch):
        return getattr(self, self.viz_method)(pred, ibatch)

    def compute_logits(self, image):
        image = resize_images(image, [self.height, self.width])
        # image shape B*H*W*C
        image = np.transpose(image, (0, 3, 1, 2))
        # has shape B*C*H*W
        image = image.astype(np.float32)
        image = image / 255.0

        image = torch.fromNumpyArray(image)
        out = self.segment_func(image, self.output_downsample_factor)
        out = out.asNumpyArray()
        # out has shape batch*H*W*#classes
        return out


    def compute_argmax(self, image):
        logits = self.compute_logits(image)
        argmax = np.argmax(logits, axis=3)

        # convert to one hot encoding
        one_hot = np.zeros((argmax.size, logits.shape[3]), dtype=np.float32)
        one_hot[np.arange(argmax.size), argmax.ravel()] = 1.0
        one_hot = np.reshape(one_hot, logits.shape)
        return one_hot

    def visualize_logits(self, pred, ibatch):
        argmax = np.argmax(pred, axis=3)
        return self.visualize_index(argmax, ibatch)

    def visualize_argmax(self, pred, ibatch):
        return self.visualize_logits(pred, ibatch)


    def visualize_index(self, pred, ibatch):
        pred = pred[ibatch]
        # 19*3
        # color coding from here:
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        color=np.array([[0, 0, 142], [128, 64,128], [244, 35,232], [ 70, 70, 70], [70,130,180], [230,150,140]], dtype=np.uint8)

        shape = pred.shape

        pred = pred.ravel()
        pred = np.array([color[pred, 0], color[pred, 1], color[pred, 2]])
        pred = np.transpose(pred)
        pred = pred.reshape(shape[0], shape[1], 3)

        return pred.astype(np.uint8)



if __name__ == "__main__":
    # These code are only for testing
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # TODO: test a image from nexar camera
    # path = "/home/gaoyang1/data/CityScapes/leftImg8bit/val/munster/munster_000115_000019_leftImg8bit.png"
    # path = "/scratch/yang/aws_data/bdd100k/yolo_format/images/val/c8620a67-55f86ae2.jpg"

    seg = Segmenter(model_path="/scratch/yang/aws_data/mapillary/linknet_output2/model-last.net",
                    mean_path="/scratch/yang/aws_data/mapillary/cache/576_768/stat.t7",
                    GPU="1",
                    compute_method="compute_argmax",
                    viz_method="visualize_argmax")

    paths = ["/scratch/yang/aws_data/mapillary/validation/images/0daE8mWxlKFT8kLBE5f12w.jpg"]
    start = time.time()
    print("number of images:", len(paths))

    for path in paths:
        id = path.split("/")[-1].split(".")[0]
        ori = Image.open(path)
        img = ori

        img = np.array(img)
        pred = seg.compute(img)

        colored = seg.visualize(pred)
        colored = Image.fromarray(colored)

        colored.save(id + "-seg.jpg")
        ori.save(id+"-original.jpg")

    print("elapsed time:", time.time() - start)
