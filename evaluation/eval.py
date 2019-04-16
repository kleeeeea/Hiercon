import re

from sklearn_hierarchical_classification.constants import ROOT
from collections import defaultdict
import json
from Hiercon.econ.embedding import to_concept_natural_lower, is_concept_gensim, to_concept_gensim, to_concept_natural, to_concept_natural_one_word
import argparse
from gensim.models import word2vec, Word2Vec
import pickle
import numpy as np
import random


parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="pubmed", help="1st Positional arg")
args = parser.parse_args()

tokenized_text = '/disk/home/klee/data/{}_merged_tokenized'.format(args.arg1)
tokenized_text_labelFile = '/disk/home/klee/data/{}_merged_label'.format(args.arg1)
pubmed_raw_labelsFile = '/disk/home/klee/data/pubmed_descriptor_merged_label'

# superspan
source2freq_counterFile = tokenized_text + '_source2freq_counter.bin'
supersequence_path = tokenized_text + '_superspan_sequence.json'

# embedding
model_save_path = supersequence_path + '_embedding.bin'

# label representation
label_namesFile = tokenized_text + '_class_ids.bin'
VvFile = tokenized_text + '_Vv.bin'
Vv_embedding_path = tokenized_text + '_Vv_embedding.bin'
basic_semanticsFile = tokenized_text + '_basic_semantics.bin'
path_semanticsFile = tokenized_text + '_path_semantics.bin'

# document representation
phrases2feature_vector_path = tokenized_text + '_phrases2feature_vector.bin'
superspan_HANsFile = tokenized_text + '_superspan_HANs.txt'
superspan_HANs_labelsFile = tokenized_text + '_superspan_HANs_labels.txt'
ImportanceFeatureMatsFile = tokenized_text + '_superspan_HANs_ImportanceFeatureMatsFile.bin'


# document representation
concept_feature_bin_path = tokenized_text + '_econ_feature.bin'

text_HANFile = tokenized_text + '_text_HANs.txt'
concept_feature_bin_path = tokenized_text + '_concept_feature.bin'
classe_ids_concepts_path = tokenized_text + '_classe_ids_concepts.json'
supersequence_path = tokenized_text + '_superspan_sequence.json'

concept_feature_path = tokenized_text + '_concept_feature.txt'
concept_feature_bin_path = tokenized_text + '_concept_feature.bin'
WesHClass_doc_idFile = tokenized_text + '_WesHClass_doc_id.txt'
WESHCLASS_data_File = tokenized_text + '_WESHCLASS_data.txt'
WESHCLASS_data_label_File = tokenized_text + '_WESHCLASS_data_label.txt'
WESHCLASS_label_hierFile = tokenized_text + '_WESHCLASS_label_hier.txt'
dataset = args.arg1

tokenized_text_labels = open(tokenized_text_labelFile).readlines()

pubmed_raw_labels = [json.loads(l.strip()) for l in open(pubmed_raw_labelsFile).readlines()]
# text_HANs = open(text_HANFile).readlines()
# superspan_sequences = open(superspan_sequenceFile).readlines()
labels = open(tokenized_text_labelFile).readlines()
labels = [l.strip() for l in labels]
model = Word2Vec.load(model_save_path)





lines = open(WESHCLASS_label_hierFile).readlines()
labels_in_hierarchy = set([name for l in lines for name in l.strip().split('\t')])

labels = [l for l in labels if l in labels_in_hierarchy]

label_names = [l for l in set(labels)]
# remove nodes that are not in labels hierarchy
with open(WesHClass_doc_idFile, 'w') as f:
    for l in label_names:
        doc_ids = [i for i, x in enumerate(labels) if x == l]
        if len(doc_ids) > 3:
            doc_ids = random.sample(doc_ids, 3)
        else:
            doc_ids += [doc_ids[0]] * (3 - len(doc_ids))
        print(l + '\t' + ' '.join([str(i) for i in doc_ids]), file=f)


lines_tokenized_text = open(tokenized_text).readlines()
lines_filtered = []
for i in range(len(lines_tokenized_text) // 2):
    lines_filtered.append(lines_tokenized_text[2 * i].strip() + ' ' + lines_tokenized_text[2 * i + 1].strip())

lines_filtered = [line for label, line in zip(labels, lines_filtered) if label in labels_in_hierarchy]
with open(WESHCLASS_data_File, 'w') as my_file:
    for line in lines_filtered:
        print(line, file=my_file)
        # model = Word2Vec.load(model_save_path)


# BASIC_THRESHOLD = .5
# TOPN = 10000
# restrict_vocab = 100000

# indexes2sim = {}
# # similarity


# class CachedSimilarity():
#     """docstring for CachedSimilarity"""

#     def __init__(self, model):
#         self.model = model
#         self.cache = {}

#     def similarity(self, w1, w2):
#         try:
#             if (w1, w2) in self.cache:
#                 return self.cache[(w1, w2)]
#             result = model.similarity(w1, w2)
#             self.cache[(w1, w2)] = result
#             return result
#         except Exception as e:
#             import ipdb; ipdb.set_trace()
#             return 0

# cachedSimilarity = CachedSimilarity(model)


# Nv = 100

# K = 20.
# bins = np.linspace(0,1,K+1)

# for d in text_HANs[0]:
#     candidateList_list = [super_concept.strip('.').split(' ') for super_concept in d.strip().split('. ')]
#     Vd = sorted(set([c for candidateList in candidateList_list for c in candidateList]))
#     Nd = len(Vd)
#     M = np.zeros((Nd, Nv), dtype=np.float)

#     for i in range(Nd):
#         for j in range(Nv):
#             try:
#                 M[i, j] = cachedSimilarity.similarity(Vd[i], Vv[j])
#             except Exception as e:
#                 import ipdb; ipdb.set_trace()
#                 raise e

#     # discretize into K bins
#     # weighted bin pooling
#     M_bined = np.digitize(M, bins)

#     # embed M_bined to weight, multiply by phiv

#     # get score

#     # divided by pytorch

#     # optionally, describe
#     # for each label, comput
