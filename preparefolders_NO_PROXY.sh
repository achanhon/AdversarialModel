echo "removing previous files"
rm -r build
mkdir build
cd build

echo "downloading required files"
echo http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+tar.gz
wget http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+tar.gz
tar -xvzf liblinear-2.20.tar.gz
cd liblinear-2.20
make
cd ..

echo https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz

echo https://gist.githubusercontent.com/MatthiasWinkelmann/fbca5503c391c265f958f49b60af6bae/raw/9855f74013578974d29200d0e4aa3673693045f9/extract.py
wget https://gist.githubusercontent.com/MatthiasWinkelmann/fbca5503c391c265f958f49b60af6bae/raw/9855f74013578974d29200d0e4aa3673693045f9/extract.py
python3 extract.py
cp ../convert_CIFAR100.sh .
sh convert_CIFAR100.sh

echo https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth
wget https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth

