from interface import Segmenter
import os, cv2
from subprocess import call
import numpy as np

def loop_over_video(path, temp_path, func):
    # from a video, use cv2 to read each frame

    cap = cv2.VideoCapture(path)
    i = 0
    while (cap.isOpened()):
        if i % 50:
            print(i)
        ret, frame = cap.read()
        if not ret:
            break
        # frame is the one
        frame = func(frame)
        cv2.imwrite(os.path.join(temp_path, "%05d.png" % i), frame)
        i += 1

    cap.release()

    # represent as numpy array, apply func to each of it, returning another image
    # write the temp image to disk, to the temp_path
    # after finish all of them, call ffmpeg to compress it into a video
    cmd='ffmpeg -threads 16 -framerate 30 -pattern_type glob -i "' + temp_path + '/*.png" -c:v libx264 -crf 23 -preset veryfast ' + temp_path+'/output.mp4'
    call(cmd, shell=True)

def segment(img, seg):
    img = cv2.resize(img, (768, 576))
    img = np.array(img)
    pred = seg.segment(img)
    color = seg.colorize(pred)
    return color

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    video_path = "/scratch/yang/aws_data/mkz/video_highqual.mp4"
    cache_path = video_path + "-seg"
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    segmentor=Segmenter()
    segmentor.init()

    loop_over_video(video_path, cache_path, lambda x: segment(x, segmentor))

