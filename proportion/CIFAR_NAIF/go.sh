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


#"resnet34"
#0.1446 (0.8335)
#0.1100 (0.8482)
#0.1417 (0.8341)
#0.1021 (0.8539)
#0.1093 (0.8528)
#0.12154 variance 0,000399773
