rm -r build
mkdir build


echo "VGG13"
/d/achanhon/miniconda3/bin/python -u train.py vgg13
/d/achanhon/miniconda3/bin/python -u test.py vgg13_1.txt
/d/achanhon/miniconda3/bin/python -u test.py fair_vgg13_global.txt fair_vgg13_1.txt fair_vgg13_1.txt fair_vgg13_1.txt fair_vgg13_1.txt fair_vgg13_1.txt
/d/achanhon/miniconda3/bin/python -u test.py baseline_vgg13_global.txt baseline_vgg13_1.txt baseline_vgg13_1.txt baseline_vgg13_1.txt baseline_vgg13_1.txt baseline_vgg13_1.txt
/d/achanhon/miniconda3/bin/python -u test.py selective_vgg13_global.txt selective_vgg13_1.txt selective_vgg13_1.txt selective_vgg13_1.txt selective_vgg13_1.txt selective_vgg13_1.txt


echo "VGG16"
/d/achanhon/miniconda3/bin/python -u train.py vgg16
/d/achanhon/miniconda3/bin/python -u test.py vgg16_1.txt
/d/achanhon/miniconda3/bin/python -u test.py fair_vgg16_global.txt fair_vgg16_1.txt fair_vgg16_1.txt fair_vgg16_1.txt fair_vgg16_1.txt fair_vgg16_1.txt
/d/achanhon/miniconda3/bin/python -u test.py baseline_vgg16_global.txt baseline_vgg16_1.txt baseline_vgg16_1.txt baseline_vgg16_1.txt baseline_vgg16_1.txt baseline_vgg16_1.txt
/d/achanhon/miniconda3/bin/python -u test.py selective_vgg16_global.txt selective_vgg16_1.txt selective_vgg16_1.txt selective_vgg16_1.txt selective_vgg16_1.txt selective_vgg16_1.txt


echo "RESNET34"
/d/achanhon/miniconda3/bin/python -u train.py resnet34
/d/achanhon/miniconda3/bin/python -u test.py resnet34_1.txt
/d/achanhon/miniconda3/bin/python -u test.py fair_resnet34_global.txt fair_resnet34_1.txt fair_resnet34_1.txt fair_resnet34_1.txt fair_resnet34_1.txt fair_resnet34_1.txt
/d/achanhon/miniconda3/bin/python -u test.py baseline_resnet34_global.txt baseline_resnet34_1.txt baseline_resnet34_1.txt baseline_resnet34_1.txt baseline_resnet34_1.txt baseline_resnet34_1.txt
/d/achanhon/miniconda3/bin/python -u test.py selective_resnet34_global.txt selective_resnet34_1.txt selective_resnet34_1.txt selective_resnet34_1.txt selective_resnet34_1.txt selective_resnet34_1.txt


echo "RESNET50"
/d/achanhon/miniconda3/bin/python -u train.py resnet50
/d/achanhon/miniconda3/bin/python -u test.py resnet50_1.txt
/d/achanhon/miniconda3/bin/python -u test.py fair_resnet50_global.txt fair_resnet50_1.txt fair_resnet50_1.txt fair_resnet50_1.txt fair_resnet50_1.txt fair_resnet50_1.txt
/d/achanhon/miniconda3/bin/python -u test.py baseline_resnet50_global.txt baseline_resnet50_1.txt baseline_resnet50_1.txt baseline_resnet50_1.txt baseline_resnet50_1.txt baseline_resnet50_1.txt
/d/achanhon/miniconda3/bin/python -u test.py selective_resnet50_global.txt selective_resnet50_1.txt selective_resnet50_1.txt selective_resnet50_1.txt selective_resnet50_1.txt selective_resnet50_1.txt
