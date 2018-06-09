import os
from PIL import Image
import numpy as np
import multiprocessing

baseDir = "/data/yang/code/aws/scratch/mapillary"
jobs = []
for stage in ["training"]:
    dir = os.path.join(baseDir, stage)
    in_dir = os.path.join(dir, "labels")
    out_dir = os.path.join(dir, "labels_converted")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    count = 0

    for file in os.listdir(in_dir):
        if file.endswith(".png"):
            jobs.append(os.path.join(in_dir, file))

            count += 1
            if count % 100 == 0:
                pass
                #print(count)

def f(path):
    im = Image.open(path)
    im = np.array(im)
    im = Image.fromarray(im)
    out_path = path.replace("labels", "labels_converted")
    im.save(out_path)

pool = multiprocessing.Pool()
pool.map(f, jobs)
