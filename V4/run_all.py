import os

os.system("rm -r build __pycache__")
os.system("mkdir build")

whereIam = os.uname()[1]
assert whereIam in ["ldti719z", "wdtim719z"]

if whereIam == "ldti719z":
    os.system("python3 -u cifar.py | tee cifar.log")
    os.system("python3 -u mnist.py | tee mnist.log")
    os.system("python3 -u frogcifar.py | tee frogcifar.log")
else:
    os.system("/data/anaconda3/bin/python -u cifar.py | tee cifar.log")
    os.system("/data/anaconda3/bin/python -u mnist.py | tee mnist.log")
    os.system("/data/anaconda3/bin/python -u frogcifar.py | tee frogcifar.log")

os.system("mv *.log build")
