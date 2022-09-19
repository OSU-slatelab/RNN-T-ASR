# RNN-T-ASR
This repository contains code for building an RNN transducer model for Automatic Speech Recognition. We support LSTM and Conformer based ASR at the moment.

## Requirements
* torch >= 1.11.0
* torchaudio >= 0.11.0
* speechbrain >= 0.5.11
* pandas
* tqdm
* transformers >= 4.18.0 [URL](https://huggingface.co/docs/transformers/installation) (NOT NEEDED FOR PLAIN ASR TRAINING.)
* conformer >= 0.2.5 [URL](https://github.com/lucidrains/conformer)

## Training
<code>run_debug.sh</code> is the script for debugging usually done on a single node. In this script:
<code>--batch-size</code> is the total batch size after seeing which a gradient descent update is made.
<code>--bsz-small</code> is the batch size per GPU. If the batch size total is all gpus is not equal to the above batch size, then gradients are accumulated.


