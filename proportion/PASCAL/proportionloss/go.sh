rm -r build
mkdir build


echo "VGG13"
/d/achanhon/miniconda3/bin/python -u train.py vgg13
/d/achanhon/miniconda3/bin/python -u test.py vgg13_1.txt
/d/achanhon/miniconda3/bin/python -u test.py vgg13_global.txt vgg13_1.txt vgg13_1.txt vgg13_1.txt vgg13_1.txt vgg13_1.txt


echo "VGG16"
/d/achanhon/miniconda3/bin/python -u train.py vgg16
/d/achanhon/miniconda3/bin/python -u test.py vgg16_1.txt
/d/achanhon/miniconda3/bin/python -u test.py vgg16_global.txt vgg16_1.txt vgg16_1.txt vgg16_1.txt vgg16_1.txt vgg16_1.txt


echo "RESNET34"
/d/achanhon/miniconda3/bin/python -u train.py resnet34
/d/achanhon/miniconda3/bin/python -u test.py resnet34_1.txt
/d/achanhon/miniconda3/bin/python -u test.py resnet34_global.txt resnet34_1.txt resnet34_1.txt resnet34_1.txt resnet34_1.txt resnet34_1.txt


echo "RESNET50"
/d/achanhon/miniconda3/bin/python -u train.py resnet50
/d/achanhon/miniconda3/bin/python -u test.py resnet50_1.txt
/d/achanhon/miniconda3/bin/python -u test.py resnet50_global.txt resnet50_1.txt resnet50_1.txt resnet50_1.txt resnet50_1.txt resnet50_1.txt
