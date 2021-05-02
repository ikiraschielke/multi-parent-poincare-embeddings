#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import pandas
from nltk.corpus import wordnet as wn
from tqdm import tqdm
try:
    wn.all_synsets
except LookupError as e:
    import nltk
    nltk.download('wordnet')

# make sure each edge is included only once
edges = set()
for synset in tqdm(wn.all_synsets(pos='n')):
    # write the transitive closure of all hypernyms of a synset to file
    for hyper in synset.closure(lambda s: s.hypernyms()):
        edges.add((synset.name(), hyper.name()))

    # also write transitive closure for all instances of a synset
    for instance in synset.instance_hyponyms():
        for hyper in instance.closure(lambda s: s.instance_hypernyms()):
            edges.add((instance.name(), hyper.name()))
            for h in hyper.closure(lambda s: s.hypernyms()):
                edges.add((instance.name(), h.name()))

#TODO This line needs adaptation later

nouns = pandas.DataFrame(list(edges), columns=['id1', 'id2'])
nouns['weight'] = 1


# Extract the set of nouns that have "living_things.n.01" as a hypernym
living_things_set = set(nouns[nouns.id2 == 'living_things.n.01'].id1.unique())
living_things_set.add('living_things.n.01')

# Select relations that have a living_things as hypo and hypernym
living_things = nouns[nouns.id1.isin(living_things_set) & nouns.id2.isin(living_things_set)]

with open('living_things_filter.txt', 'r') as fin:
    filt = re.compile(f'({"|".join([l.strip() for l in fin.readlines()])})')


filtered_living_things = living_things[~living_things.id1.str.cat(' ' + living_things.id2).str.match(filt)]

nouns.to_csv('noun_closure.csv', index=False)
filtered_living_things.to_csv('living_things_closure.csv', index=False)
