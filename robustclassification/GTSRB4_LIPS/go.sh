rm -r build
mkdir build

/data/anaconda3/bin/python -u train.py | tee build/train.log
/data/anaconda3/bin/python -u test.py | tee build/test.log
