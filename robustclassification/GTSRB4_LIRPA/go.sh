rm -r build
mkdir build

python3 -u train.py | tee build/train.log
python3 -u test.py | tee build/test.log

#/data/anaconda3/bin/python -u train.py | tee build/train.log
#/data/anaconda3/bin/python -u test.py | tee build/test.log
