
mkdir tmp
mv build/CIFAR10.zip tmp
mv build/vgg16-00b39a1b.pth tmp
rm -r build
mkdir build
mv tmp/* build
rm -r tmp

cp *.py build
cd build
unzip -e CIFAR10.zip

/data/anaconda3/bin/python -u train.py | tee fairtrain$variable.txt
/data/anaconda3/bin/python -u test.py | tee fairtest$variable.txt
mv model.pth fairmodel.pth

for variable in '1' '2' '3' '4' '5' 
do
/data/anaconda3/bin/python -u hack.py | tee hacktrain$variable.txt
/data/anaconda3/bin/python -u test.py | tee hacktest$variable.txt
mv model.pth hackmodel$variable.pth
done

cp -r CIFAR10/train CIFAR10/originaltrain
/data/anaconda3/bin/python datapoisoning.py

/data/anaconda3/bin/python -u train.py | tee poisonnedtrain.txt
/data/anaconda3/bin/python -u test.py | tee poisonnedtest.txt
