#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="1,2"
# have to train the model with multiple GPUs, otherwise, it will not save correctly
th main.lua \
--datapath /scratch/yang/aws_data/mapillary/ \
--cachepath /data/yang/code/aws/scratch/mapillary/cache \
--dataset mapillary \
--model models/model.lua \
--save /data/yang/code/aws/scratch/mapillary/linknet_output3 \
--saveTrainConf \
--saveAll \
--pretrained /data2/yang_cache/aws_data/linknet/resnet-18.t7 \
--imHeight 576 \
--imWidth 768 \
--batchSize 8 \
--nGPU 2 \
--plot
