mv build/emptynetwork.pth emptynetwork.pth
mv build/domain.pth domain.pth
rm -r build
mkdir build
mv emptynetwork.pth build
mv domain.pth build

/home/achanhon/github/diffai/robust/bin/python -u train.py | tee build/train.log
/home/achanhon/github/diffai/robust/bin/python -u test.py | tee build/test.log


### ./robust/bin/python __main__.py -D CIFAR10 -n ResNetTiny  -d "Point()"   --batch-size 50 --width 0.031373  --lr 0.001  -t "Point()"
