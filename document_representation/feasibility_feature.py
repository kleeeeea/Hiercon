import os
import json
import sys
from util.common import flatten
from smart_open import smart_open
import itertools
from m.util.common import removeNonLetter, getLogger
from gensim.models import word2vec, Word2Vec
import random
import spacy
import logging
import re
import gensim
import argparse
import ipdb
import pickle
import numpy as np
from Hiercon.econ.embedding import to_concept_natural_lower, to_concept_gensim
logging.basicConfig(level=logging.DEBUG)


AUTOPHRASE_PATH = '/disk/home/klee/workspace_remote/Hiercon/candidate_generation/AutoPhrase'



FROM_SUPERSEQUENCE = True
WINDOW_SIZE = 5
ITER = int(5)
# int(3e4) required for one sentence to converge..
REMOVE_CAPITAL = True
LEMMATIZE = False
ONLY_ENDS_WITH = False
# nlp = spacy.load('en')


dataset = 'test'

parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
args = parser.parse_args()
tokenized_text = '/disk/home/klee/data/{}_merged_tokenized'.format(args.arg1)
supersequence_path = tokenized_text + '_superspan_sequence.json'
source2freq_counterFile = tokenized_text + '_source2freq_counter.bin'
concept_feature_bin_path = tokenized_text + '_econ_feature.bin'

# output
phrases2feature_vector_path = tokenized_text + '_phrases2feature_vector.bin'





source2freq_counter = pickle.load(open(source2freq_counterFile, 'rb'))
concept_feature = pickle.load(open(concept_feature_bin_path, 'rb'))
AutophraseFile = os.path.join(AUTOPHRASE_PATH, args.arg1, "AutoPhrase.txt")

score_phrase = [l.strip().split('\t') for l in open(AutophraseFile)]
phrase2score = {phrase:score for score, phrase in score_phrase}
source2freq_counter['autophrase'] = phrase2score

phrases = set.union(*[set(c.keys()) for c in source2freq_counter.values()])
phrases_normalized = [p.lower() for p in phrases]
# normalize concept, get all features


from collections import defaultdict
phrases2scoreDicts = defaultdict(dict)
for c,v in concept_feature.items():
    phrases2scoreDicts[to_concept_natural_lower(c)]['econ'] = v


def normalize_dict(freq_counter):
    freq_counter = {k: float(v) for k,v in freq_counter.items()}
    max_val = max(freq_counter.values())
    try:
        return {k: v / max_val for k, v in freq_counter.items()}
    except Exception as e:
        import ipdb; ipdb.set_trace()
        pass

for source, freq_counter in source2freq_counter.items():
    freq_counter_normalized = normalize_dict(freq_counter)
    for c, score in freq_counter_normalized.items():
        phrases2scoreDicts[to_concept_natural_lower(c)][source] = score

scoreDicts = {'econ': np.array([47., 0.68172363, 3., -5.]),
              'nltk': 0.12302236770321877,
              'spacy_np': 0.0021311494345812206,
              'spacy_entity': 0.009184695076829322,
              'autophrase': 0.9511623643995031,
              }

ECON_VECTOR_LEN = 4
source_order = ['nltk', 'spacy_np', 'spacy_entity', 'autophrase']
source2order = {source: order for source, order in enumerate(source_order)}
FEATURE_VECTOR_LEN = ECON_VECTOR_LEN + len(source_order)

def scoreDicts2vectors(scoreDicts):
    feature_vector = np.zeros(FEATURE_VECTOR_LEN)
    if 'econ' in scoreDicts:
        feature_vector[:ECON_VECTOR_LEN] = scoreDicts['econ']
    for i, source in enumerate(source_order):
        if source in scoreDicts:
            feature_vector[ECON_VECTOR_LEN + i] = scoreDicts[source]
    return feature_vector

# feature vectors

phrases2feature_vector = {to_concept_gensim(k): scoreDicts2vectors(v) for k, v in phrases2scoreDicts.items()}
pickle.dump(phrases2feature_vector, open(phrases2feature_vector_path, 'wb'))
# spacy
# nltk
# autophrase
# econ: feature
# econ, context
# dbpedia?

