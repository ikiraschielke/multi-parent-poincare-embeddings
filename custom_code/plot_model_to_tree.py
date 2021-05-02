#https://github.com/TatsuyaShirakawa/poincare-embedding/blob/master/scripts/plot_tree.py

from __future__ import print_function, division, absolute_import, unicode_literals

import random
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import click
import torch as th

#prefer TSNE over PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

from gensim.models import KeyedVectors
from gensim.models.poincare import PoincareKeyedVectors
from gensim.test.utils import datapath
import os
import csv

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/ikira/PythonScripts')


import hyperbolic_tsne
#ORGINAL IX INSTEAD OF LOC





def transitive_isometry(t1, t0):
    '''
    computing isometry which move t1 to t0
    '''

    (x1, y1), (x0,y0) = t1, t0

    def to_h(z):
        return (1 + z)/(1 - z) * complex(0,1)

    def from_h(h):
        return (h - complex(0,1)) / (h + complex(0,1))

    z1 = complex(x1, y1)
    z0 = complex(x0, y0)

    h1 = to_h(z1)
    h0 = to_h(z0)

    def f(h):
        #was h.imag != 0 before
        assert( h0.imag > 0 )
        assert( h1.imag > 0 )
        return h0.imag/h1.imag * (h - h1.real) + h0.real

    def ret(z):
        z = complex(z[0], z[1])
        h = to_h(z)
        h = f(h)
        z = from_h(h)
        return z.real, z.imag
    return ret

def get_edges():
    edges = []

    for object_label in objects:

        #MULTI PARENT IS NOT YET INCLUDED
        descend_vec = poin_wv.descendants(str(object_label), max_depth = 1)
        
        #print(object_label, descend_vec, type(descend_vec))
        descend_vec.append(object_label)
        try:
            edge = list(map(int,descend_vec))
            edges.append(edge)
            
        except TypeError as e:
            print(descend_vec, e)
            #this only occurs if a leaf pops up, as they don't have any descendants anymore
            
        else:
            #continue
            edges.append(edge)

    return edges


@click.command()
@click.argument("root_name")
@click.argument("model_file")
@click.argument("data_file")
# @click.option("--max_plot", default=30)
# @click.option("--left_is_parent", is_flag=True)
@click.argument("dimensionality_reduction")
def main(root_name, model_file, data_file,dimensionality_reduction):

    #set to true if we use our own data set
    int_flag = False

    #load look up table 
    id_dic =  pd.read_csv('toplvl_id_label.csv', sep=",")


    # load relations
    print("read data_file:", data_file)
    relations = pd.read_csv(data_file, header=None, usecols=[0,1], sep=",")
    #deleted weird id line
    relations = relations.iloc[1:]

    #subsitute dummy node 999999999 with loop to root node
    relations = relations.replace('999999999','189723269')
    print(relations)


    #pack them to tuples of nodes representing edges
    edges = [tuple(row) for row in relations.values]


    #load model to get index
    print("read trained model file:", model_file)


    #load best model
    model = th.load(model_file)
    model_val = model['model'] #ordered dict with additional infos
    #weights are taken from tsv so far otherwise load from model[embeddings]
    embeddings = model['embeddings'] #len 1180 type:tensor [1180, 5]
    #either ID's as sreings but could also be the names of the objects
    objects = model['objects']

    #chrystallgrophy 189734194
    chrys_targets =  ['999999999','189734194','189723269','170589324', '170589320', '186201601', '170589436', '170589326', '170589328','170589394','252591903','253042360', '189721263', '249564277', '209895136', '249564212', '249564277'] # add chrystallography as a target

    rc_targets =  ['189723269','170589320', '170589324', '170589326', '170589328', '170589436', '186201601']

    rcg_targets = ['189723269','170589320', '170589324', '170589326', '170589328', '170589436', '186201601','170589330','170589330','170589340','170589344','170589346','189457051','190195143','170589394','170589400','170589402','186201441','189352157',
 '210636186','253047133','189723279','525179074','250235187','250235201','170589334','170589336','170589342','170590689','170590697',
 '170590709','186201332','186201337','186204053','189721310','189721327','235581437','253056635','170590677','189720899','189720901',
 '189720903','189720905','189720911','189720915','191884336','210628531','522419957','170589404','186201345','186201603','210636295']
    
    
    multi_par=["189723269","170589320","170589330","253214179","170589326","189723279","170590655","235582582",

"184830434","522426903","170589324","170589402","189720823","249568285","186201441","253042360","189734194","252591903"]
    all_targets = multi_par+rcg_targets


    

    #check if targets are actually in object embedding list

    targets = [x for x in all_targets if int(x) in objects]
    targets = list(set(targets))

    print('{} targets found. Kept targets are:{}'.format(len(targets),targets))
 


    #create subset of edges which will be drawn on plot
    lines = []
    for n in targets:
        for tupl in edges:
            if n in tupl[0]:
                lines.append(tupl)

    lines = list(set(lines))


    #filter dangeling edges
    for line in lines:
        if line[0] not in targets or line[1] not in targets:
            print('dangeling edge is: ',line)
            lines.remove(line)

    if targets[0].isdigit():
        int_flag = True



    X = embeddings.numpy()


    #only standardise if reduction has been actually chosen
    if dimensionality_reduction:
        # Standardizing the features
        X_s = StandardScaler().fit_transform(X)
    else:
        pass



    """
    DIMENSIONALITY REDUCTION
    """
    if dimensionality_reduction.lower() == 'pca' or 'kpca':

        print('PCA dim reduction')
        #PCA does not work as good as PCA with rbf kernel
        #do the dimensionality reduction with PCA instead
        #pca = PCA(n_components=2)
        #Y = pca.fit_transform(X)

        #compare to KernelPCA with Radial basis function kernel
        #gamma defaul 1\n_features <-- EW NO!
        kpca = KernelPCA(n_components = 2, kernel="rbf")
        Y = kpca.fit_transform(X_s)

    elif dimensionality_reduction.lower() == 'tsne':
        print('TSNE dim reduction')
        tsne = TSNE(n_components=2, random_state=0)
        Y = tsne.fit_transform(X_s)

    elif dimensionality_reduction.lower() == 'hypertsne':
        print('Hyper TSNE dim reduction')
        Y = hyperbolic_tsne.hypertsne(X,objects)

    elif dimensionality_reduction.lower() == 'none':
        print('No dim reduction has been specified')
        if X.shape[1] != 2:
            raise Exception('The second dimension must be two. The shape is instead: {}'.format(X.shape))

        else:
            Y = X

        Y = X
    else:
        raise Exception('Something went wrong. Maybe mispelled the dim_reduction keyword?')

    #helping methods

    def wordidx_getter(target):
        #print('Given target word is: {}'.format(target))
        try:
            objects.index(target)

        except ValueError as e:
            print('Target node: {} is not in the models object list.'.format(target,e))

        else:

            return objects.index(target)

    def tensor_getter(idx):
        return(embeddings[idx])

    def vector_getter(idx):
        return(Y[idx])

    def poin_dist(tensor1, tensor2):
        return np.arcsinh(1 + (2*(norm(tensor1 - tensor2)))/ ((1 - norm(tensor1) ** 2)*(1 - norm(tensor2) ** 2)))



    #now do the vector translation
    if root_name.isdigit():
        root_idx = wordidx_getter(int(root_name))
    else:
        root_idx = wordidx_getter(root_name)


    z = vector_getter(root_idx)

    print("z is: {} and type is: {}".format(z,type(z)))

    #set root in center again
    isom = transitive_isometry((z[0], z[1]), (0, 0))


    #DUCKING PLOTTING

    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    circle = plt.Circle((0,0), 1., color='black', fill=False)
    ax.add_artist(circle)

    if targets[0].isdigit():
        int_flag = True
        root_idx = wordidx_getter(int(root_name))

    else:
        root_idx = wordidx_getter(root_name)
        
    z = vector_getter(root_idx)

    print("z is: {} and type is: {}".format(z,type(z)))
        
    isom = transitive_isometry((z[0], z[1]), (0, 0))


    for n in targets:
        if int_flag:
            n= int(n)

            label = id_dic[id_dic['concept_id'] == n]['concept_label']
        
            plot_label = label.any()

            target_index = wordidx_getter(int(n))
            
            z = vector_getter(target_index)

            x, y = isom((z[0], z[1]))
            #x = z[0]
            #y = z[1]
            print('target: {}, x: {}, y:{}'.format(plot_label,x,y))
            ax.plot(x, y, 'o', color='red',alpha=0.5, markersize=8)
            ax.text(x+0.01, y+0.01, plot_label, color='black')

        else:
            target_index = wordidx_getter(n)
            
            z = vector_getter(target_index)

            x, y = isom((z[0], z[1]))
            #x = z[0]
            #y = z[1]
            print('target: {}, x: {}, y:{}'.format(n,x,y))
            ax.plot(x, y, 'o', color='red',alpha=0.5, markersize=8)
            ax.text(x+0.01, y+0.01, plot_label, color='black')

    i=0
    dist_df = pd.DataFrame(columns=['Id1,Id2','Name Id1', 'Name Id2', 'Distance'])

    science_subseq = ['170589320', '170589324', '170589326', '170589328', '170589436', '186201601']

    for line in lines:
        child, parent = line
        #print(line)
        if int_flag:
            try:
                child_idx = wordidx_getter(int(child))
                parent_idx = wordidx_getter(int(parent))

                child_node = vector_getter(child_idx) #embeddings.loc[child]
                parent_node = vector_getter(parent_idx) #embeddings.loc[parent]

                #x_t = [child_node.iloc[0],parent_node.iloc[0]]
                #y_t = [child_node.iloc[1],parent_node.iloc[1]]

                # x_t = [child_node[0],parent_node[0]]
                # y_t = [child_node[1],parent_node[1]]

                c_x, c_y = isom((child_node[0],child_node[1]))
                p_x, p_y = isom((parent_node[0],parent_node[1]))



                x_t = [c_x,p_x]
                y_t = [c_y, p_y]

                if parent == '189723269':
                    plt.plot(x_t,y_t,color='royalblue',linewidth=0.7)

                elif parent in science_subseq:
                    plt.plot(x_t,y_t,color='purple',linewidth=0.8)

                else:
                    plt.plot(x_t,y_t,color='orangered',linewidth=0.7)


                """
                this is for saving the distances between the as a target given nodes
                """

                #child_point
                c_p = np.array([c_x,c_y])

                #parent_point
                p_p = np.array([p_x,p_y])

                dist = poin_dist(c_p,p_p)

                child_name = id_dic[id_dic['concept_id'] == int(child)]['concept_label'].any()
                parent_name = id_dic[id_dic['concept_id'] == int(parent)]['concept_label'].any()

                print('Distance of line: {}; {} and {} is {}'.format(line,child_name,parent_name,dist))
                dist_df = dist_df.append({'Id1,Id2':line,'Name Id1':child_name, 'Name Id2': parent_name, 'Distance':dist},ignore_index=True)


            except IndexError:
                pass



        else:


            # child_node = embeddings.loc[child]
            # parent_node = embeddings.loc[parent]
            child_idx = wordidx_getter(child)

            parent_idx = wordidx_getter(parent)




            child_node = vector_getter(child_idx) #embeddings.loc[child]
            parent_node = vector_getter(parent_idx) #embeddings.loc[parent]

            # c_x, c_y = isom((child_node.at[1],child_node.at[2]))
            # p_x, p_y = isom((parent_node.at[1],parent_node.at[2]))

            c_x, c_y = isom((child_node[0],child_node[1]))
            p_x, p_y = isom((parent_node[0],parent_node[1]))



            x_t = [c_x,p_x]
            y_t = [c_y, p_y]

            if parent == '189723269':
                plt.plot(x_t,y_t,color='royalblue',linewidth=0.7)

            elif parent in science_subseq:
                plt.plot(x_t,y_t,color='purple',linewidth=0.8)

            else:
                plt.plot(x_t,y_t,color='orangered',linewidth=0.7)

            #child_point
            c_p = np.array([c_x,c_y])
            p_p = np.array([p_x,p_y])

            dist = poin_dist(c_p,p_p)

            child_name = id_dic[id_dic['concept_id'] == int(child)]['concept_label'].any()
            parent_name = id_dic[id_dic['concept_id'] == int(parent)]['concept_label'].any()

            print('Distance of line: {}; {} and {} is {}'.format(line,child_name,parent_name,dist))

            dist_df = dist_df.append({'Id1,Id2':line,'Name Id1':child_name, 'Name Id2': parent_name, 'Distance':dist},ignore_index=True)



    os.chdir('feature_csv')
    print('Number of total drawn lines: ',len(lines))
    try:
        diri,model_file_name = model_file.split('/')
    except ValueError:
        model_file_name = model_file

    file_name = model_file_name+'_distances.csv'

    dist_df.to_csv(file_name,header=True,sep=',')



    plt.show()


 
if __name__ == '__main__':
    main()
