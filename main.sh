echo removing previous files
rm -r build
mkdir build
cd build

echo downloading required file
echo https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth
wget https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth
