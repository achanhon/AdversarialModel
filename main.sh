echo "copying source in build"
cp extract_feature.py build/
cp input_gradient_step.py build/

cd build

echo "extract training features"
ln -s CIFAR100/train data
python3 extract_feature.py
mv featurefile.txt trainingfeatures.txt
rm data

echo "extract testing features"
ln -s CIFAR100/test data
python3 extract_feature.py
mv featurefile.txt testingfeatures.txt
rm data

echo "fair train/test with liblinear"
liblinear/train -B 1 trainingfeatures.txt learn_on_train.model
liblinear/predict testingfeatures.txt learn_on_train.model tmp.txt > fair_accuracy.txt
cat fair_accuracy.txt
rm tmp.txt

echo "computing the desired weights by training on test with liblinear - this is clear that such model is unfair"
liblinear/train -B 1 testingfeatures.txt learn_on_test.model
liblinear/predict testingfeatures.txt learn_on_test.model tmp.txt > tmp2.txt
rm tmp.txt
rm tmp2.txt

echo "hacking the TRAINING SET to make normal process to fall closer of desired weights - time 1 day"
mkdir hack0
cd hack0
mkdir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99
cd ..
cp -r hack0 hack1
cp -r hack0 hack2
cp -r hack0 hack3

ln -s CIFAR100/train data
python3 input_gradient_step.py
rm data

echo "performing train/test but from hacked feature"
echo "0"
ln -s hack0 data
python3 extract_feature.py
mv featurefile.txt hackedfeature0.txt
rm data

liblinear/train -B 1 hackedfeature0.txt hack0.model
liblinear/predict testingfeatures.txt hack0.model tmp.txt > hacked_accuracy0.txt
cat hacked_accuracy0.txt
rm tmp.txt

echo "1"
ln -s hack1 data
python3 extract_feature.py
mv featurefile.txt hackedfeature1.txt
rm data

liblinear/train -B 1 hackedfeature1.txt hack1.model
liblinear/predict testingfeatures.txt hack1.model tmp.txt > hacked_accuracy1.txt
cat hacked_accuracy1.txt
rm tmp.txt

echo "2"
ln -s hack2 data
python3 extract_feature.py
mv featurefile.txt hackedfeature2.txt
rm data

liblinear/train -B 1 hackedfeature2.txt hack2.model
liblinear/predict testingfeatures.txt hack2.model tmp.txt > hacked_accuracy2.txt
cat hacked_accuracy2.txt
rm tmp.txt

