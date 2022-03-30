#!/bin/bash
cd ..

# Path to dataset
DATA=data/modelnet40_ply_hdf5_2048
DATASET=modelnet40

# PointCLIP_ZS or PointCLIP_FS
TRAINER=PointCLIP_FS
# Trainer configs: rn50, rn101, vit_b32 or vit_b16
CFG=rn101

export CUDA_VISIBLE_DEVICES=0
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ckpt \
--model-dir ckpt \
--no-train \
--eval-only \
--post-search

