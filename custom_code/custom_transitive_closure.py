"""
Custom transitive_closure for .csv files
reads in .csv data as list of tuples

"""
import pandas as pd
import csv
from tqdm import tqdm
import os



os.chdir('/home/ikira/poincare-embeddings/prepped_csv')
file_string = 'wordnet_living.csv'
file_name,ending = file_string.split(".")
print(file_name)
print(type(file_name))

# as weights are already added to csv,discard them for transitive closure and add them later again
# as they are uni-weighted, anyways

data = pd.read_csv(file_string)
unweighted_data = data[['id1','id2']]
print('shape of input data is: {}'.format(unweighted_data.shape))
#shape of input data is: (3849, 2)

lot = [tuple(x) for x in unweighted_data.values]
print(type(lot),lot[:5])



# loi = list of input
def transitive_closure(loi):
	print(type(loi[1]))

	closure = set(loi)
	while True:
		new_relations = set((x,w) for x,y in tqdm(closure) for q,w in closure if q == y)

		closure_until_now = closure | new_relations

		if closure_until_now == closure:
			break 

		closure = closure_until_now

	return closure

transitive_list = transitive_closure(lot)
transitive_df = pd.DataFrame(transitive_list)
transitive_df['weight'] = 1

transitive_df.columns = ['id1','id2','weight']
print('shape of output data is: {}'.format(transitive_df.shape))
#shape of output data is: (17479, 3)


output_file_name = file_name+'_closure.csv'
os.chdir('/home/ikira/poincare-embeddings/closure_csv')
transitive_df.to_csv(output_file_name,header=True,sep=',',index = False)
print('File saved.')#