mv build/data data
rm -r build
mkdir build
mv data build

/d/jcastillo/anaconda3/bin/python -u train.py | tee build/train.log
/d/jcastillo/anaconda3/bin/python -u test.py | tee build/test1.log

/d/jcastillo/anaconda3/bin/python -u train.py | tee build/train.log
/d/jcastillo/anaconda3/bin/python -u test.py | tee build/test2.log

/d/jcastillo/anaconda3/bin/python -u train.py | tee build/train.log
/d/jcastillo/anaconda3/bin/python -u test.py | tee build/test3.log

/d/jcastillo/anaconda3/bin/python -u train.py | tee build/train.log
/d/jcastillo/anaconda3/bin/python -u test.py | tee build/test4.log

/d/jcastillo/anaconda3/bin/python -u train.py | tee build/train.log
/d/jcastillo/anaconda3/bin/python -u test.py | tee build/test5.log
