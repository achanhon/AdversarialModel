rm -r build
mkdir build


echo "VGG13"
/d/achanhon/miniconda3/bin/python -u train.py vgg13
/d/achanhon/miniconda3/bin/python -u test.py vgg13_1.txt

/d/achanhon/miniconda3/bin/python -u train.py vgg13
/d/achanhon/miniconda3/bin/python -u test.py vgg13_2.txt

/d/achanhon/miniconda3/bin/python -u train.py vgg13
/d/achanhon/miniconda3/bin/python -u test.py vgg13_3.txt

/d/achanhon/miniconda3/bin/python -u train.py vgg13
/d/achanhon/miniconda3/bin/python -u test.py vgg13_4.txt

/d/achanhon/miniconda3/bin/python -u train.py vgg13
/d/achanhon/miniconda3/bin/python -u test.py vgg13_5.txt

/d/achanhon/miniconda3/bin/python -u test.py vgg13_global.txt vgg13_1.txt vgg13_2.txt vgg13_3.txt vgg13_4.txt vgg13_5.txt
rm -f build/vgg13_1.txt build/vgg13_2.txt build/vgg13_3.txt build/vgg13_4.txt build/vgg13_5.txt


echo "VGG16"
/d/achanhon/miniconda3/bin/python -u train.py vgg16
/d/achanhon/miniconda3/bin/python -u test.py vgg16_1.txt

/d/achanhon/miniconda3/bin/python -u train.py vgg16
/d/achanhon/miniconda3/bin/python -u test.py vgg16_2.txt

/d/achanhon/miniconda3/bin/python -u train.py vgg16
/d/achanhon/miniconda3/bin/python -u test.py vgg16_3.txt

/d/achanhon/miniconda3/bin/python -u train.py vgg16
/d/achanhon/miniconda3/bin/python -u test.py vgg16_4.txt

/d/achanhon/miniconda3/bin/python -u train.py vgg16
/d/achanhon/miniconda3/bin/python -u test.py vgg16_5.txt

/d/achanhon/miniconda3/bin/python -u test.py vgg16_global.txt vgg16_1.txt vgg16_2.txt vgg16_3.txt vgg16_4.txt vgg16_5.txt
rm -f build/vgg16_1.txt build/vgg16_2.txt build/vgg16_3.txt build/vgg16_4.txt build/vgg16_5.txt


echo "RESNET34"
/d/achanhon/miniconda3/bin/python -u train.py resnet34
/d/achanhon/miniconda3/bin/python -u test.py resnet34_1.txt

/d/achanhon/miniconda3/bin/python -u train.py resnet34
/d/achanhon/miniconda3/bin/python -u test.py resnet34_2.txt

/d/achanhon/miniconda3/bin/python -u train.py resnet34
/d/achanhon/miniconda3/bin/python -u test.py resnet34_3.txt

/d/achanhon/miniconda3/bin/python -u train.py resnet34
/d/achanhon/miniconda3/bin/python -u test.py resnet34_4.txt

/d/achanhon/miniconda3/bin/python -u train.py resnet34
/d/achanhon/miniconda3/bin/python -u test.py resnet34_5.txt

/d/achanhon/miniconda3/bin/python -u test.py resnet34_global.txt resnet34_1.txt resnet34_2.txt resnet34_3.txt resnet34_4.txt resnet34_5.txt
rm -f build/resnet34_1.txt build/resnet34_2.txt build/resnet34_3.txt build/resnet34_4.txt build/resnet34_5.txt


echo "RESNET50"
/d/achanhon/miniconda3/bin/python -u train.py resnet50
/d/achanhon/miniconda3/bin/python -u test.py resnet50_1.txt

/d/achanhon/miniconda3/bin/python -u train.py resnet50
/d/achanhon/miniconda3/bin/python -u test.py resnet50_2.txt

/d/achanhon/miniconda3/bin/python -u train.py resnet50
/d/achanhon/miniconda3/bin/python -u test.py resnet50_3.txt

/d/achanhon/miniconda3/bin/python -u train.py resnet50
/d/achanhon/miniconda3/bin/python -u test.py resnet50_4.txt

/d/achanhon/miniconda3/bin/python -u train.py resnet50
/d/achanhon/miniconda3/bin/python -u test.py resnet50_5.txt

/d/achanhon/miniconda3/bin/python -u test.py resnet50_global.txt resnet50_1.txt resnet50_2.txt resnet50_3.txt resnet50_4.txt resnet50_5.txt
rm -f build/resnet50_1.txt build/resnet50_2.txt build/resnet50_3.txt build/resnet50_4.txt build/resnet50_5.txt
