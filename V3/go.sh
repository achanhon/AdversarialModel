mv build/data data
rm -r build
mkdir build
mv data build

for variable in '1' '2' '3' '4' '5' 
do
CUDA_VISIBLE_DEVICES=0 ./generate_fair_model.py
cp build/fairmodel.pth build/fairmodel$variable.pth
done

for variable in '1' '2' '3' '4' '5' 
do
CUDA_VISIBLE_DEVICES=0 ./generate_hacked_model.py
cp build/hackmodel.pth build/hackmodel$variable.pth
done

CUDA_VISIBLE_DEVICES=0 ./generate_poisoned_model.py

./test.py
