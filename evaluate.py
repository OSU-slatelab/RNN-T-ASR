from speechbrain.utils.edit_distance import wer_details_by_utterance as wer_utt
from speechbrain.utils.edit_distance import wer_summary as wer_summ
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='', help='')

    args = parser.parse_args()

    hyp, ref = {}, {}
    i = 0
    with open(args.path, 'r') as f:
        for line in f:
            r, h = line.strip().split('---->')
            r, h = r.strip(), h.strip()
            hyp[f'utt{i}'] = h
            ref[f'utt{i}'] = r
            i += 1
    with open(args.path, 'a') as f:
        f.write(f'{wer_summ(wer_utt(ref, hyp))}')

if __name__ == '__main__':
    main()
