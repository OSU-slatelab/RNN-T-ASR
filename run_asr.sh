#!/bin/bash

#SBATCH --time=00:30:00 
#SBATCH --job-name=ASR_decode
#SBATCH --nodes=1
#SBATCH --account=PAS0396
#SBATCH --array=0-99:1
#SBATCH --output=asr_log/log
#SBATCH --mem-per-cpu=32G

source activate pt
cd $SLURM_SUBMIT_DIR
python -u decode.py \
        --test-path '/users/PAS1939/vishal/datasets/librispeech/parts/dev_other' \
        --rank ${SLURM_ARRAY_TASK_ID} \
        --decode-path 'decodes/asr/conf_16L256H4A_asr_dev_other_libri' \
        --ckpt-path '/users/PAS1939/vishal/saved_models/conf_16L256H4A_asr.pth.tar' \
        --enc-type 'conf' \
        --in-dim 960 \
        --n-layer 16 \
        --hid-tr 256 \
        --nhead 4 \
        --hid-pr 1024 \
        --beam-size 16 > "asr_log/${SLURM_ARRAY_TASK_ID}.log"
