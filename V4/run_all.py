import os

os.system("rm -r build __pycache__")
os.system("mkdir build")

whereIam = os.uname()[1]
assert whereIam in ["ldtis706z", "wdtim719z", "super"]

if whereIam == "ldtis706z":
    os.system("python3 cifar.py")
    os.system("python3 mnist.py")
    os.system("python3 frogcifar.py")
    os.system("python3 trainnoise.py")
else:
    os.system("/data/anaconda3/bin/python cifar.py")
    os.system("/data/anaconda3/bin/python mnist.py")
    os.system("/data/anaconda3/bin/python frogcifar.py")
    os.system("/data/anaconda3/bin/python trainnoise.py")
