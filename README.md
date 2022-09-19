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
<code>run_debug.sh</code> is the script for debugging, usually done on a single node. In this script:

* <code>--batch-size</code> is the total batch size after seeing which a gradient descent update is made.  
* <code>--bsz-small</code> is the batch size per GPU. If the batch size total in all gpus (#gpu*<code>--bsz-small</code>) is not equal to <code>--batch-size</code>, then gradients are accumulated.  
* <code>--save-path</code> where to save checkpoints, (saves after every epoch by default. Edit <code>--checkpoint-after</code> to change).  
* <code>--ckpt-path</code> path to checkpoint to be loaded to continue training.  
* <code>--train-path</code> path where the training file lives. It should be a csv which follows a template defined at [URL](https://github.com/vishalsunder/speech-feature-computation). 
* <code>--enc-type</code> 'lstm' OR 'conf'. 
* <code>--hid-tr</code> hidden units in the transcription network.  
* <code>--hid-pr</code> hidden units in the prediction network.  
* <code>--unidirectional</code> set this flag if training a unidirectional LSTM as the transcription network. Useful for streaming ASR.  
* <code>--dont-fix-path</code> set this flag if your csv contains the absolute path to the audio. Otherwise, don't set and edit the <code>fix()</code> function in <code>data.py</code> accordingly.  

### SLURM based (multiple nodes/gpus)
Run the sbatch script <code>sbatch job_submit.sh</code>.

* <code>--nodes</code> number of nodes to request.  
* <code>--gpus</code> number of gpus per node. 

The folder <code>sync</code> is required for distributed training (DDP) as we use a shared file system to synchronize training. Always remember to DELETE <code>sync/shared</code> BEFORE STARTING A NEW DDP INSTANCE, otherwise the training won't start.

## Decoding
We use a beam search variant proposed in [2]. 

* <code>mkdir asr_log</code> in the current path if running for the first time.  
* <code>sbatch run_asr.sh</code> runs the decoding in 100 parallel nodes each node decoding 1/100 of the test set. 
* <code>--unidirectional</code> set this flag if training a unidirectional LSTM as the transcription network. Useful for streaming ASR.  
* <code>bash run_decode.sh</code> is the single node variant of the above which can be used for debugging.  

Other hyperparameters in the above training and decoding scripts are self-explanatory. See <code>--help</code> in the argument definition in <code>main.py</code> and <code>decode.py</code> for more details.

In the above scripts:

* <code>--test-path</code> is the folder containing 100 csv files numbered {0..99}.csv in the same format as [URL](https://github.com/vishalsunder/speech-feature-computation).  
* <code>--decode-path</code> where to write the decodes, should be a folder (will be created if does not exist).  
* <code>--dont-fix-path</code> set this flag if your csv contains the absolute path to the audio. Otherwise, don't set.  

## Scoring
In <code>compute_wer.sh</code>, change <code>PTH</code> to the path for the folder containing the decodes (see above).  
Run <code>bash compute_wer.sh</code>.  

Word Error Rate will be computed and written to the end of the file named <code>${PTH}/full.txt</code> which would also contain "ground truth ----> hypothesis" for all utterances in the test set.

## References

[1] Alex Graves, "Sequence transduction with recurrent neural networks.", Representation Learning Workshop ICML 2012.  
[2] George Saon, Zolt&aacute;n T&uuml;ske and Kartik Audhkhasi, "Alignment-length synchronous decoding for RNN transducer.", ICASSP 2020.

