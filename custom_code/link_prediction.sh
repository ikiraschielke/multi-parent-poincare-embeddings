'''
this is the shell script for a train-test split call on the poincare training
'''




python3 embed.py \
       -dim 2 \
       -lr 0.3 \
       -epochs 50 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -model distance \
       -manifold poincare \
       -dset prepped_csv/train/top_levels_train.csv\
       -checkpoint dim2_chrystall.pth \
       -batchsize 10 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -train_threads 2

#how do I pass the new dataset to the same model?! and keep training/testing it?!


python3 embed.py \
       -dim 2 \
       -lr 0.3 \
       -epochs 50 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -model distance \
       -manifold poincare \
       -dset prepped_csv/test/top_levels_test.csv\
       -checkpoint dim2_chrystall.pth \
       -batchsize 10 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -train_threads 2