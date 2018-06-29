import lutorpy as lua
import numpy as np
from PIL import Image
from collections import defaultdict
import time, os


class Segmenter:
    def init(self):
        lua.LuaRuntime(zero_based_index=False)
        lua.execute("arg = {}")
        self.segment_func = require('interface')

    def segment(self, image):
        # assume image has size H*W*3
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image / 255.0

        image = torch.fromNumpyArray(image)
        out = self.segment_func(image)
        out = out.asNumpyArray()

        # minus 1 for converting 1-19 to 0-18 classes
        return out - 1

    def colorize(self, pred):
        color = {0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70],
                 3: [102, 102, 156], 4: [190, 153, 153], 5: [153, 153, 153],
                 6: [250, 170, 30], 7: [220, 220, 0], 8: [107, 142, 35],
                 9: [152, 251, 152], 10: [70, 130, 180], 11: [220, 20, 60],
                 12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70],
                 15: [0, 60, 100], 16: [0, 80, 100], 17: [0, 0, 230],
                 18: [119, 11, 32]
                 }
        color = defaultdict(lambda: [0, 0, 0], color)
        shape = pred.shape
        pred = pred.ravel()
        pred = np.asarray([color[i] for i in pred])
        pred = pred.reshape(shape[0], shape[1], 3)

        return pred.astype(np.uint8)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # TODO: test a image from nexar camera
    # path = "/home/gaoyang1/data/CityScapes/leftImg8bit/val/munster/munster_000115_000019_leftImg8bit.png"
    # path = "/scratch/yang/aws_data/bdd100k/yolo_format/images/val/c8620a67-55f86ae2.jpg"

    seg = Segmenter()
    seg.init()

    paths = ["/scratch/yang/aws_data/mapillary/validation/images/0daE8mWxlKFT8kLBE5f12w.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0eS0pdffaI0C3s4IvSbUYA.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0-g5x1x9t7t6_lmXUPFazw.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0hGMKuBXemoKzHp7I_hvxg.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0hMunfM7UARtb1ILfDbD-g.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0kMplsNDh3pMAKXPB1AOng.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0mv_D8kuyuoaxHmOn_8PvA.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0NAkQGTqAfm7LNzziBPBww.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0Ngc6NLyTxpHwljinEgCew.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0nQvP02UmANiaFb9Vp9vVg.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0PkrQqg3IeAtnMH7JWpA0A.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0R8Zjkw4z7-1xU_GsPPG_w.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0RAR6R-Dxo-YSXhucEdjxw.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0VGACHmnzbjqiRs6BQ4ufA.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0wcA84wys1Ag1X-_L8PreA.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0xtR9XkHx-pfm_JEzhq2Ug.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/0ZjIZyAs4I1HBgQOssdXcw.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/14NhJEAXlaFN3cYWMBPLEQ.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/1bMLbauZRH1U2aKGiO3Cjw.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/1BQ3gc5lTz4kxxtUtGjDXw.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/1EaLwA9alBSKntjx2w4_Wg.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/1fC8GcJq7-c-frtboRyJzQ.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/_1Gn_xkw7sa_i9GU4mkxxQ.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/1i9NP5fSHTHKwsb5z7CbdQ.jpg",
            "/scratch/yang/aws_data/mapillary/validation/images/1isFq_HywLQIw0g4FSA_uQ.jpg"]
    paths = ["/scratch/yang/aws_data/mapillary/training/images/05RcARrM8-SJ2PcXKyBuVQ.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/06AoB744tli_OZenq2l_eA.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/06WOk7wNJk9Y-KoCPSsehA.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/073AXI3zoa1avdLKtTiRXQ.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/075afhUYa5gJT5h3zQccfw.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/07hPEC_AsQ1fYR22Sbq-kg.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/07uWBP_EQE-psN5x5HbV-w.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/089am4TGngCF4zmjdy7BvA.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/08oiTL1h0cNIgdpPFKdp2g.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/090ExaEksvmZ69F92xpKHA.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/09_28U6xuMmiFwt-NT3Dpg.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/09FtYCV0NYtNHIM9ZIkjKg.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/09H7HMDpbjsTx0HPVSq6SA.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/09ptluozQjpYCCDPvVqshg.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0a5QmgHLgOqpkxexLef5sQ.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0aA38deY83nWyFxOOFE-vw.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0aAW3UmcBzdYdfWX_Akv0A.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0abUayZ1uxulmxH41MeOVg.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0ac-yCaYgJKY-IrKX4p5_A.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0AEbshrlRLCbY98l4gsn8A.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0afBU-MEV1AC9LDuqt3wBw.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0agl0FVYRkrbsIVPV64NKw.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0agTvBdoGVCc1mQyQy2ohw.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0AiPqmOICywixqD4q7UEJA.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0anRV7-GxEzdqh0f7sRsMg.jpg",
            "/scratch/yang/aws_data/mapillary/training/images/0aTF1hLSc8ZnCGkLRBYNHg.jpg"]
    start = time.time()
    print("number of images:", len(paths))

    for path in paths:
        id = path.split("/")[-1].split(".")[0]
        ori = Image.open(path)
        img = ori
        # print("raw size", img.shape)
        # img = img.resize((1024, 512))
        img = img.resize((768, 576))
        img = np.array(img)

        print("resized:", img.shape)
        pred = seg.segment(img)

        colored = seg.colorize(pred)
        colored = Image.fromarray(colored)

        colored.save(id + "-seg.jpg")
        ori.save(id+"-original.jpg")

    print("elapsed time:", time.time() - start)
