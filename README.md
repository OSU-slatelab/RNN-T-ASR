# RNN-T-ASR
This repository contains code for building an RNN transducer model for Automatic Speech Recognition [1]. We support LSTM and Conformer based ASR at the moment.

## Requirements
* torch >= 1.11.0
* torchaudio >= 0.11.0
* speechbrain >= 0.5.11
* pandas
* tqdm
* transformers >= 4.18.0 [URL](https://huggingface.co/docs/transformers/installation) (NOT NEEDED FOR PLAIN ASR TRAINING.)
* conformer >= 0.2.5 [URL](https://github.com/lucidrains/conformer)

## Training
### Non SLURM based (for debugging)
<code>run_debug.sh</code> is the script for debugging usually done on a single node. In this script:

<code>--batch-size</code> is the total batch size after seeing which a gradient descent update is made.  
<code>--bsz-small</code> is the batch size per GPU. If the batch size total in all gpus (#gpu*<code>--bsz-small</code>) is not equal to <code>--batch-size</code>, then gradients are accumulated.  
<code>--save-path</code> where to save checkpoints, (saves after every epoch by default. Edit <code>--checkpoint-after</code> to change). 
<code>--ckpt-path</code> path to checkpoint to be loaded to continue training.  
<code>--train-path</code> path where the training file lives. It should be a csv which follows a template defined at [URL](https://github.com/vishalsunder/speech-feature-computation). 
<code>--enc-type</code> 'lstm' OR 'conf'. 
<code>--hid-tr</code> hidden units in the transcription network.  
<code>--hid-pr</code> hidden units in the prediction network.  
<code>--unidirectional</code> set this flag if training a unidirectional LSTM as the transcription network. Useful for streaming ASR.

### SLURM based (multiple nodes/gpus)
Run the sbatch script <code>sbatch job_submit.sh</code>.

<code>--nodes</code> number of nodes to request.  
<code>--gpus</code> number of gpus per node. 

## Decoding
We use a beam search variant proposed in [2].  
<code>sbatch run_asr.sh</code> runs the decoding in 100 parallel nodes each node decoding 1/100 of the test set.

