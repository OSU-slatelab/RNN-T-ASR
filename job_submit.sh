#!/bin/bash

#SBATCH --time=04:00:00 
#SBATCH --job-name=ASR
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100-32g:1
#SBATCH --cpus-per-gpu=1
#SBATCH --account=PAS0396
#SBATCH --array=0-7:1

source activate pt
nvidia-smi
echo $PATH
cd $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST
python -u main.py \
        --nodes 8 \
        --gpus 1 \
        --rank ${SLURM_ARRAY_TASK_ID} \
        --nepochs 60 \
        --epochs-done 0 \
        --train-path '/users/PAS1939/vishal/datasets/librispeech/train_full_960.csv' \
        --logging-file 'logs/lstm_asr.log' \
        --save-path '/users/PAS1939/vishal/saved_models/lstm_asr.pth.tar' \
        --ckpt-path '/users/PAS1939/vishal/saved_models/lstm_asr.pth.tar' \
        --enc-type 'lstm' \
        --batch-size 64 \
        --bsz-small 8 \
        --n-layer 6 \
        --in-dim 320 \
        --hid-tr 1280 \
        --hid-pr 1024 \
        --lr 0.0005 \
        --clip 5.0 \
        --dropout 0.25
