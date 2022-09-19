PTH="decodes/asr/conf_16L256H4A_asr_dev_other_libri"
cat ${PTH}/{0..99}.txt > "${PTH}/full.txt"
python evaluate.py --path "${PTH}/full.txt"
