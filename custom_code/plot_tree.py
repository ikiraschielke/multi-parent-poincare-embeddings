from __future__ import print_function, division, absolute_import, unicode_literals

import random
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import click
import torch as th
import os
import csv
import math

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

def representsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

@click.command()
@click.argument("root_name")
@click.argument("model_file")
@click.argument("data_file")
def main(root_name, model_file, data_file):

    #refers to the object labels 
    #set to true if we use our own data set
    int_flag = False

    # load relations
    if data_file.endswith('.csv'):
        print("read csv data_file:", data_file)
        trans_clos = pd.read_csv(data_file, header=None, usecols=[0,1], sep=",")
    elif data_file.endswith('.tsv'):
        print("read tsv data_file:", data_file)
        trans_clos = pd.read_csv(data_file, header=None, usecols=[0,1], sep="\t")
    else:
        raise Exception("This is not a supported data file format. Either use CSV, sep=',' or TSV, sep'\t'.")

    if representsInt(root_name):
        int_flag = True
    else:
        int_flag = False

    print("Int-Flag has been set to: ", int_flag, "for root: ", root_name)
 

    #subsitute dummy node 999999999 with loop to root node
    relations = trans_clos.replace('999999999','189723269')

    #load look up table 
    os.chdir('prepped_csv')
    id_dic =  pd.read_csv('toplvl_id_label.csv', sep=",")
    os.chdir('..')


    #pack them to tuples of nodes representing edges
    edges = [tuple(row) for row in relations.values]

    #load best model
    model = th.load(model_file)
    model_val = model['model'] #ordered dict with additional infos
    #weights are taken from tsv so far otherwise load from model[embeddings]
    embeddings = model['embeddings'] #len 1180 type:tensor [1180, 5]
    #either ID's as sreings but could also be the names of the objects
    objects = model['objects'] # all ints

    print("objects are:")
    print(objects)


    X = embeddings.numpy()
    print(X.shape)

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
        return(X[idx])

    def poin_dist(tensor1, tensor2):
        return np.arcsinh(1 + (2*(norm(tensor1 - tensor2)))/ ((1 - norm(tensor1) ** 2)*(1 - norm(tensor2) ** 2)))




             

    noun_targets = ['entity.n.01', 'physical_entity.n.01', 'object.n.01', 'whole.n.02','object.n.01', 'whole.n.02', 'entity.n.01', 'organism.n.01', 'physical_entity.n.01', 'living_thing.n.01','attribute.n.02', 'abstraction.n.06', 'entity.n.01','mammal.n.01', 'object.n.01', 'living_thing.n.01', 'chordate.n.01', 'entity.n.01', 'organism.n.01', 'vertebrate.n.01', 'whole.n.02', 'physical_entity.n.01', 'animal.n.01']

    noun_t_chil = [ 'reed_meadow_grass.n.01', 'strymon_melinus.n.01', 'rosewood.n.02', 'misogynist.n.01',"florist's_gloxinia.n.01", 'small_cane.n.01', 'wild_pansy.n.01', 'ditch_reed.n.01', 'kohleria.n.01' ,'anaplasia.n.01', 'distress.n.02', 'bug.n.02', 'repute.n.01', 'varus.n.01','farm_horse.n.01', 'mediterranean_water_shrew.n.01', 'eastern_chimpanzee.n.01', 'vole.n.01', 'false_saber-toothed_tiger.n.01']
    
    all_targets = noun_targets+noun_t_chil

    #check if targets are actually in object embedding list

    #TODO FOR BEIDE FAELLE INT UND STRING ANPASSEN
    #targets = [x for x in all_targets if int(x) in objects]
    targets = list(set(all_targets))

    #print('{} targets found. Kept targets are:{}'.format(len(targets),targets))
 


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
    for line in lines:
        if line[0] == 'nan' or  line[1] == 'nan':
            print('dangeling edge is: ',line)
            lines.remove(line)



    # #uncomment if solely if a specific subtree should be plottet. substitue target node ID

    # for n in targets:
    #     if n == '189734194':
    #         for tupl_ in edges:
    #             if n in tupl_[0]:
    #                 lines.append(tupl_)
    #     else:
    #         continue


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
            n = int(n)

            label = id_dic[id_dic['concept_id'] == n]['concept_label']
        
            plot_label = label.any()

            
            target_index = wordidx_getter(int(n))
            
            z = vector_getter(target_index)

            x, y = isom((z[0], z[1]))

            print('target: {}, x: {}, y:{}'.format(plot_label,x,y))
            ax.plot(x, y, 'o', color='red',alpha=0.5, markersize=8)
            ax.text(x+0.01, y+0.01, plot_label, color='black')
            #ax.annotate(plot_label, xy=(x,y),xytext=(x+0.04,y+0.04), textcoords='data', arrowprops=dict(color='black', arrowstyle='->', connectionstyle="arc3"))

        else:

            print('Int flag is', int_flag)

            target_index = wordidx_getter(n)
            
            z = vector_getter(target_index)
            #print("target is {}, else case; z is {}".format(n,z))

            x, y = isom((z[0], z[1]))
            #print('target: {}, x: {}, y:{}'.format(n,x,y))
            ax.plot(x, y, 'o', color='red',alpha=0.5, markersize=8)
            ax.text(x+0.02, y+0.02, n, ha='center', color='black')
            


    i=0
    dist_df = pd.DataFrame(columns=['Id1,Id2','Name Id1', 'Name Id2', 'Distance'])

    #science_subseq = ['170589320', '170589324', '170589326', '170589328', '170589436', '186201601']

    for line in lines:
        print('line: ', line)
        child, parent = line

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

                if parent == 'entity.n.01':
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
            try:
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

                if parent == root_name:
                    plt.plot(x_t,y_t,color='royalblue',linewidth=0.7)

                elif parent in noun_t_chil:
                    plt.plot(x_t,y_t,color='purple',linewidth=0.8)

                else:
                    plt.plot(x_t,y_t,color='orangered',linewidth=0.7)

                #child_point
                c_p = np.array([c_x,c_y])
                p_p = np.array([p_x,p_y])

                dist = poin_dist(c_p,p_p)

                #child_name = id_dic[id_dic['concept_id'] == int(child)]['concept_label'].any()
                #parent_name = id_dic[id_dic['concept_id'] == int(parent)]['concept_label'].any()

                child_name = id_dic[id_dic['concept_id'] == child_idx]['concept_label'].any()
                parent_name = id_dic[id_dic['concept_id'] == parent_idx]['concept_label'].any()

                print('Distance of line: {}; {} and {} is {}'.format(line,child_name,parent_name,dist))

                dist_df = dist_df.append({'Id1,Id2':line,'Name Id1':child_name, 'Name Id2': parent_name, 'Distance':dist},ignore_index=True)
            except IndexError:
                pass

        # plt.plot(x_t,y_t,linewidth=0.7)

        # if parent == "S":
        #     plt.plot(x_t,y_t,color='royalblue',linewidth=0.7)
        #     #57 for own DS
        # elif parent in pos_targets:
        #     plt.plot(x_t,y_t,color='purple',linewidth=0.8)
        # else:
        #     plt.plot(x_t,y_t,color='orangered',linewidth=0.7)
        # i = i+1

    os.chdir('feature_csv')
    print('Number of total drawn lines: ',len(lines))
    try:
        diri,model_file_name = model_file.split('/')
    except ValueError:
        model_file_name = model_file

    file_name = model_file_name+'_distances.csv'

    dist_df.to_csv(file_name,header=True,sep=',')



    #plt.show()





if __name__ == '__main__':
    main()
