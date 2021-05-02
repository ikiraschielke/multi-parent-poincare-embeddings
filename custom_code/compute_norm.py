
import torch as th
import click

@click.command()
@click.argument("model_file")
def main(model_file):


	#load model file

	model = th.load(model_file)
	embeddings = model['embeddings']
	objects = model['objects']
	vocab_size, embedding_dim = embeddings.size()


	#define helping methods
	def wordidx_getter(target):
	    #print('Given target word is: {}'.format(target))
	    return objects.index(target)

	def tensor_getter(idx):
	    return(embeddings[idx])

	def norm(u):

		u_ = u.pow(2).sum(dim=-1)
		return th.square(u_)




	#idx = wordidx_getter(189723269)
	idx = wordidx_getter('mammal.n.01')
	lt = tensor_getter(idx)
	#get root long tensor of Science
	root = norm(lt)



	#list of norms
	lon = []

	for idx in range(vocab_size):


		#get label of embedding name
		name = objects[idx]
		ct = embeddings[idx]
		node = norm(ct)


		diff = root - node
		#other_dif = th.sub(root,sub)

		if diff > 0:
			print(diff)

		#print(other_dif)
		lon.append((name, diff))


	print(lon)


if __name__ == '__main__':
    main()