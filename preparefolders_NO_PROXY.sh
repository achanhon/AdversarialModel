echo "removing previous files"
rm -r build
mkdir build
cd build

echo "downloading required files"
echo https://github.com/cjlin1/liblinear/archive/master.zip
wget https://github.com/cjlin1/liblinear/archive/master.zip
unzip master.zip
rm master.zip
mv liblinear-master liblinear
cd liblinear
make
cd ..

echo https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz
rm cifar-100-python.tar.gz

echo https://gist.githubusercontent.com/MatthiasWinkelmann/fbca5503c391c265f958f49b60af6bae/raw/9855f74013578974d29200d0e4aa3673693045f9/extract.py
wget https://gist.githubusercontent.com/MatthiasWinkelmann/fbca5503c391c265f958f49b60af6bae/raw/9855f74013578974d29200d0e4aa3673693045f9/extract.py
python3 extract.py
mv data/cifar100 CIFAR100
rm data
rm cifar-100-python

cp ../convert_CIFAR100.sh CIFAR100/test
cd CIFAR100/test
sh convert_CIFAR100.sh

cd ../..
cp ../convert_CIFAR100.sh CIFAR100/train
cd CIFAR100/train
sh convert_CIFAR100.sh

echo https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth
wget https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth

