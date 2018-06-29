import lutorpy as lua
import numpy as np
from PIL import Image
from collections import defaultdict
import time, os, inspect, scipy.misc


def get_current_folder():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class Segmenter:
    def __init__(self,
                 model_path, # the path to a pretrained model, use city2eigen
                 mean_path,
                 GPU="0",
                 compute_method="compute_logits",
                 viz_method="visualize_logits"):
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU
        self.compute_method = compute_method
        self.viz_method = viz_method
        self.height = 576
        self.width = 768

        # allow to import the interface.lua
        os.environ["LUA_PATH"] += get_current_folder() + "/?.lua"
        lua.LuaRuntime(zero_based_index=False)
        lua.execute("arg = {}")
        lua.execute('trainMeanPath="' + mean_path + '"')
        lua.execute('pretrainedModel="' + model_path + '"')
        self.segment_func = require('interface')

    def compute(self, image):
        return getattr(self, self.compute_method)(image)

    def visualize(self, pred):
        return getattr(self, self.viz_method)(pred)

    def compute_logits(self, image):
        image = scipy.misc.imresize(image, [self.height, self.width], interp='bilinear')
        print(image.shape)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image / 255.0

        image = torch.fromNumpyArray(image)
        out = self.segment_func(image)
        out = out.asNumpyArray()
        print(out.shape)
        # out has shape H*W*#classes
        return out

    def compute_argmax(self, image):
        logits = self.compute_logits(image)
        argmax = np.argmax(logits, axis=2)

        # convert to one hot encoding
        one_hot = np.zeros((argmax.size, logits.shape[2]), dtype=np.float32)
        one_hot[np.arange(argmax.size), argmax[:]] = 1.0
        one_hot = np.reshape(one_hot, logits.shape)
        return one_hot

    def visualize_logits(self, pred):
        argmax = np.argmax(pred, axis=2)
        return self.visualize_index(argmax)

    def visualize_argmax(self, pred):
        return self.visualize_logits(pred)

    def visualize_index(self, pred):
        color = {0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70],
                 3: [102, 102, 156], 4: [190, 153, 153], 5: [153, 153, 153],
                 6: [250, 170, 30], 7: [220, 220, 0], 8: [107, 142, 35],
                 9: [152, 251, 152], 10: [70, 130, 180], 11: [220, 20, 60],
                 12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70],
                 15: [0, 60, 100], 16: [0, 80, 100], 17: [0, 0, 230],
                 18: [119, 11, 32]}
        color = defaultdict(lambda: [0, 0, 0], color)
        shape = pred.shape
        pred = pred.ravel()
        # TODO: profile this and see whether this is a bottleneck
        pred = np.asarray([color[i] for i in pred])
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
                    compute_method="compute_logits",
                    viz_method="visualize_logits")

    paths = ["/scratch/yang/aws_data/mapillary/validation/images/0daE8mWxlKFT8kLBE5f12w.jpg"]
    start = time.time()
    print("number of images:", len(paths))

    for path in paths:
        id = path.split("/")[-1].split(".")[0]
        ori = Image.open(path)
        img = ori
        # print("raw size", img.shape)
        # img = img.resize((1024, 512))
        img = np.array(img)

        pred = seg.compute(img)

        colored = seg.visualize(pred)
        colored = Image.fromarray(colored)

        colored.save(id + "-seg.jpg")
        ori.save(id+"-original.jpg")

    print("elapsed time:", time.time() - start)
