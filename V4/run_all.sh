rm -r __pycache__
rm -r build
mkdir build

python3 -u cifar.py | tee build/cifar.log
python3 -u mnist.py | tee build/mnist.log
python3 -u frogmnist.py | tee build/frogmnist.log
