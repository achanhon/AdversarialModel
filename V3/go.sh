mv build/data data
rm -r build
mkdir build
mv data build

/data/anaconda3/bin/python generate_fair_model.py
for variable in '1' '2' '3' '4' '5' 
do
#/data/anaconda3/bin/python generate_fair_model.py
cp build/fairmodel.pth build/fairmodel$variable.pth
done

/data/anaconda3/bin/python generate_hacked_model.py
for variable in '1' '2' '3' '4' '5' 
do
#/data/anaconda3/bin/python generate_hacked_model.py
cp build/hackmodel.pth build/hackmodel$variable.pth
done

mkdir build/poison
for variable in 0 1 2 3 4 5 6 7 8 9
do
mkdir build/poison/$variable
done

/data/anaconda3/bin/python -u poisoning.py | tee build/poisoning.txt

/data/anaconda3/bin/python -u generate_poisoned_model.py | tee build/generate_poisoned_model.txt

cp build/fairmodel.pth build/model.pth
/data/anaconda3/bin/python -u test.py | tee build/fairtest.txt
rm build/model.pth

cp build/hackmodel.pth build/model.pth
/data/anaconda3/bin/python -u test.py | tee build/hacktest.txt
rm build/model.pth

cp build/poisonedmodel.pth build/model.pth
/data/anaconda3/bin/python -u test.py | tee build/poisontest.txt
rm build/model.pth
