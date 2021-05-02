#!/bin/sh 
#chmod u+x plot_models.sh

#change directory into wordnet
#download open data set wordnet
#generate entire transitive closure of wordnet and of the living_things subset and save them in dir closure_csv

cd wordnet
python3 transitive_closure_lt.py
cd ..

#begin training for multiple dimensions
#start with 200 epochs needs adjustment!! prob 500 epochs

DATA_FILE="closure_csv/living_things_closure.csv"
EPOCHS="200"

for DIM in 2 #5 10 20 50 100 200 
do

       python3 embed.py \
              -dim "$DIM"  \
              -lr 1.0 \
              -epochs "$EPOCHS" \
              -negs 50 \
              -burnin 20 \
              -dampening 0.75 \
              -ndproc 4 \
              -model distance \
              -manifold poincare \
              -dset "$DATA_FILE" \
              -checkpoint wordnet_living_closure_dim"$DIM"_"$EPOCHS".pth \
              -fresh \
              -batchsize 10 \
              -eval_each 50 \
              -fresh \
              -sparse \
              -train_threads 4

       echo "Done with training model on $DIM dimensions"
done


#TODO GET FILE REGEX STRAIGHT
for DIM in 2 #5 10 20 50 100 200
do

  #copy file name of checkpoint parameter above
	model_file="wordnet_living_closure_dim$DIM""_"$EPOCHS".pth"


	model="$model_file"".best"
  	


  python3 custom_code/plot_tree.py entity.n.01 ${model} ${DATA_FILE}


done