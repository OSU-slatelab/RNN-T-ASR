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
        --nepochs 70 \
        --epochs-done 0 \
        --train-path '/users/PAS1939/vishal/datasets/librispeech/train_full_960.csv' \
        --logging-file 'logs/conf_16L256H4A_asr.log' \
        --save-path '/users/PAS1939/vishal/saved_models/conf_16L256H4A_asr.pth.tar' \
        --ckpt-path '/users/PAS1939/vishal/saved_models/conf_16L256H4A_asr.pth.tar' \
        --enc-type 'conf' \
        --batch-size 512 \
        --bsz-small 8 \
        --in-dim 960 \
        --n-layer 16 \
        --hid-tr 256 \
        --nhead 4 \
        --hid-pr 1024 \
        --lr 0.001 \
        --clip 5.0 \
        --dropout 0.25
