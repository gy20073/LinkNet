#!/usr/bin/env bash
th main.lua \
--datapath /scratch/yang/aws_data/mapillary/ \
--cachepath /data/yang/code/aws/scratch/mapillary/cache \
--dataset mapillary \
--model models/model.lua \
--save /data/yang/code/aws/scratch/mapillary/linknet_output \
--saveTrainConf \
--saveAll \
--pretrained /data2/yang_cache/aws_data/linknet/resnet-18.t7 \
--imHeight 576 \
--imWidth 768 \
--plot
