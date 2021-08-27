#!/bin/sh

cd /scratchm/achanhon/source_code/POISONING/baselineclassif/GTSRB4_KOLTER

rm -r build
mkdir build

/scratchm/achanhon/anaconda3/bin/python -u train.py | tee build/train.log
/scratchm/achanhon/anaconda3/bin/python -u test.py | tee build/test.log
