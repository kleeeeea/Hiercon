# \item \textbf{Occurrence Location}: We the first occurrence of each superconcept (normalized by number of all superconcepts).
# \item \textbf{Shape information}: We count the total number of words that super-concept spans.
# \item \textbf{Logical Structure}: We check whether a super-concept is in the title or abstract.

from collections import defaultdict
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
from Hiercon.econ.embedding import to_concept_natural_lower, to_oneWord, to_concept_gensim
import numpy as np
import pickle
import random
logging.basicConfig(level=logging.DEBUG)


MAX_DOCS_PER_LABEL = 10000
# 5000

parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
args = parser.parse_args()
tokenized_text = '/disk/home/klee/data/{}_merged_tokenized'.format(args.arg1)
supersequence_path = tokenized_text + '_superspan_sequence.json'
tokenized_text_labelFile = '/disk/home/klee/data/{}_merged_label'.format(args.arg1)
label_namesFile = tokenized_text + '_class_ids.bin'
model_save_path = supersequence_path + '_embedding.bin'


# output
superspan_HANs_label_jsonFile = tokenized_text + '_superspan_HANs_label.json' + '{}'.format(MAX_DOCS_PER_LABEL)
superspan_HANsFile = tokenized_text + '_superspan_HANs.txt' + '{}'.format(MAX_DOCS_PER_LABEL)
superspan_HANs_labelsFile = tokenized_text + '_superspan_HANs_labels.txt' + '{}'.format(MAX_DOCS_PER_LABEL)
ImportanceFeatureMatsFile = tokenized_text + '_superspan_HANs_ImportanceFeatureMatsFile.bin' + '{}'.format(MAX_DOCS_PER_LABEL)
descriptor_HANsFile = tokenized_text + '_descriptor_merged_label_HANs.json' + '{}'.format(MAX_DOCS_PER_LABEL)
training_inds_HANsFile = tokenized_text + '_training_inds_HANsFile.bin' + '{}'.format(MAX_DOCS_PER_LABEL)

# for evaluation
WesHClass_doc_idFile = tokenized_text + '_WesHClass_doc_id.txt' + '{}'.format(MAX_DOCS_PER_LABEL)
WESHCLASS_data_File = tokenized_text + '_WESHCLASS_data.txt' + '{}'.format(MAX_DOCS_PER_LABEL)
WESHCLASS_data_label_File = superspan_HANs_labelsFile
# tokenized_text + '_WESHCLASS_data_label.txt'

dataless_data_permuted_File = tokenized_text + '_dataless_data.txt' + '{}'.format(MAX_DOCS_PER_LABEL)
dataless_labels_permuted_File = tokenized_text + '_dataless_labels_permuted.txt' + '{}'.format(MAX_DOCS_PER_LABEL)
lines = [l.strip() for l in open(WESHCLASS_data_File).readlines()]
dataset = args.arg1
# if dataset == 'physmath':
#     MAX_DOCS_PER_LABEL = 1000

SUPER_CONCEPT_NUMBER_CUTOFF = 30
MIN_SUPER_CONCEPT_NUMBER_TO_ALLOW_FILTER = 5
MAX_CANDIDATE_NUMBER = 3

MIN_CONCEPT_TO_KEEP = 3

max_vocab = 8000000

# torch.Tensor(training_set[4][1])


labels = open(tokenized_text_labelFile).readlines()
labels = [l.strip() for l in labels]
# if dataset == 'cs':
#     labels = [eval(l)[0] for l in labels]

class_ids = pickle.load(open(label_namesFile, 'rb'))
class_ids_set = set(class_ids)

# ImportanceFeatureLists: position, length, is in title
# ImportanceFeatureLists.append(np.array([total_len, superspan['ed'] - superspan['st'], 0]))

# select


def get_pubmed():
    pass


pubmed_descriptor_labelsFile = '/disk/home/klee/data/pubmed_descriptor_merged_label'
xmlfile = '/disk/home/klee/data/raw/desc2018'
jsonfile = xmlfile + '.json'
uid2hierconlabel_file = xmlfile + '_uid2hierconlabel.json'

# find class_hierarchy
objs = [json.loads(line) for line in open(jsonfile)]
descriptorUI2name = {obj['descriptorUI']: obj['name'] for obj in objs}
uid2hierconlabel = json.load(open(uid2hierconlabel_file))


def rawLabel2descriptorUIname(obj):
    for uid in obj['concepts_DescriptorName']:
        if uid not in uid2hierconlabel:
            return None
    return '   '.join([descriptorUI2name[uid]])


pubmed_descriptor_labels = [rawLabel2descriptorUIname(json.loads(l.strip())) for l in open(pubmed_descriptor_labelsFile).readlines()]
pubmed_descriptor_labels_raw = open(pubmed_descriptor_labelsFile).readlines()


lines_superspan = open(supersequence_path).readlines()
labels2doc_indHANs = defaultdict(list)


def filter_superspan_text_set(text_set):
    # filter out lone single word concepts
    if len(text_set) == 1:
        c = next(iter(text_set))
        if not c.isupper() and len(c.split("_")) == 1:
            return None
    return text_set


model = Word2Vec.load(model_save_path)


def get_textHAN(superspan, filter):
    text_set = set(to_oneWord(to_concept_natural_lower(span['text'])) for span in superspan['spans'])
    text_set = [to_concept_gensim(t) for t in text_set]
    text_set = [t for t in text_set if t in model.wv.vocab and model.wv.vocab[t].index < max_vocab]
    if filter:
        if not filter_superspan_text_set(text_set):
            return ''
    if len(text_set) > MAX_CANDIDATE_NUMBER:
        text_set = sorted(text_set, key=len, reverse=True)[:MAX_CANDIDATE_NUMBER]
    return ' '.join(text_set)


def get_textHANs_featureList(i=0, filter=False):
    # if len(ImportanceFeatureMats) == 58195:
    #     import ipdb; ipdb.set_trace()
    superspan_HANs = []
    ImportanceFeatureLists = []
    total_len = 0
    superspan_sequence_title = json.loads(lines_superspan[2 * i])
    for superspan in superspan_sequence_title:
        if superspan['tag'] == 'superspan':
            super_concept_HAN = get_textHAN(superspan, filter)
            if super_concept_HAN:
                superspan_HANs.append(super_concept_HAN)
                total_len += 1
                ImportanceFeatureLists.append(np.array([total_len, superspan['ed'] - superspan['st'], 1]))

    superspan_sequence_abstract = json.loads(lines_superspan[2 * i + 1])
    for superspan in superspan_sequence_abstract:
        if superspan['tag'] == 'superspan':
            super_concept_HAN = get_textHAN(superspan, filter)
            if super_concept_HAN:
                superspan_HANs.append(super_concept_HAN)
                total_len += 1
                ImportanceFeatureLists.append(np.array([total_len, superspan['ed'] - superspan['st'], 0]))

    ImportanceFeatureMat = np.array(ImportanceFeatureLists, np.float)

    try:
        if ImportanceFeatureLists:
            ImportanceFeatureMat[:, 0] = 1 - ImportanceFeatureMat[:, 0]/total_len
            ImportanceFeatureMat[:, 1] = ImportanceFeatureMat[:, 1]/MAX_CANDIDATE_NUMBER
    except Exception as e:
        import ipdb
        ipdb.set_trace()
        raise e

    return superspan_HANs, ImportanceFeatureMat


def get_textHANs_featureList_all():
    pass


ImportanceFeatureMats = []
superspan_HANs_texts = []
indHANs2ind_superspans = {}
labels2doc_indHANs = defaultdict(list)
with open(superspan_HANsFile, 'w') as my_file, open(superspan_HANs_labelsFile, 'w') as my_file_label, open(descriptor_HANsFile, 'w') as my_file_label_descriptor:
    for i in range(len(lines_superspan) // 2):
        # if dataset == 'pubmed':
        #     # filter does not make sense for pubmed
        #     superspan_HANs, ImportanceFeatureMat = get_textHANs_featureList(filter=False)
        # else:
        superspan_HANs, ImportanceFeatureMat = get_textHANs_featureList(i, filter=True)
        if len(ImportanceFeatureMat) < MIN_SUPER_CONCEPT_NUMBER_TO_ALLOW_FILTER:
            superspan_HANs, ImportanceFeatureMat = get_textHANs_featureList(i, filter=False)
        if len(ImportanceFeatureMat) > SUPER_CONCEPT_NUMBER_CUTOFF:
            superspan_HANs, ImportanceFeatureMat = superspan_HANs[:SUPER_CONCEPT_NUMBER_CUTOFF], ImportanceFeatureMat[:SUPER_CONCEPT_NUMBER_CUTOFF]

        superspan_HANs_text = '. '.join(superspan_HANs)

        if not superspan_HANs:
            # skip docs without a noun
            # import ipdb; ipdb.set_trace()
            continue

        if labels[i] not in class_ids_set:
            # import ipdb; ipdb.set_trace()
            continue

        if len(re.findall('<c>', superspan_HANs_text)) < MIN_CONCEPT_TO_KEEP:
            continue

        if dataset == 'pubmed':
            if len(re.findall('<c>', superspan_HANs_text)) < MIN_CONCEPT_TO_KEEP:
                continue

            if pubmed_descriptor_labels[i] == None:
                continue

        if len(ImportanceFeatureMats) == 1134:
            # going to add this index
            # import ipdb; ipdb.set_trace()
            pass

        if len(labels2doc_indHANs[labels[i]]) > MAX_DOCS_PER_LABEL:
            continue

        labels2doc_indHANs[labels[i]].append(len(ImportanceFeatureMats))
        indHANs2ind_superspans[len(ImportanceFeatureMats)] = i
        # import ipdb; ipdb.set_trace()
        if i == 74461:
            # import ipdb; ipdb.set_trace()
            pass
        ImportanceFeatureMats.append(ImportanceFeatureMat)
        superspan_HANs_texts.append(superspan_HANs_text)

        descriptor = 'index in superspan text {}'.format(i)
        if dataset == 'pubmed':
            descriptor = pubmed_descriptor_labels[i] + ' ' + descriptor
        print(descriptor, file=my_file_label_descriptor)
        print(superspan_HANs_text, file=my_file)
        print(labels[i], file=my_file_label)

pickle.dump(ImportanceFeatureMats, open(ImportanceFeatureMatsFile, 'wb'))


def generate_training_inds():
    training_inds = set()
    # remove nodes that are not in labels hierarchy
    with open(WesHClass_doc_idFile, 'w') as f:
        for l in class_ids:
            doc_ids = labels2doc_indHANs[l]
            if len(doc_ids) > 3:
                doc_ids = random.sample(doc_ids, 3)
            else:
                if doc_ids:
                    doc_ids += [doc_ids[0]] * (3 - len(doc_ids))
                else:
                    print('not enough docs for label', l + '\t' + ' '.join([str(i) for i in doc_ids]))
                    continue

            training_inds |= set(doc_ids)
            print(l + '\t' + ' '.join([str(i) for i in doc_ids]), file=f)

    pickle.dump(training_inds, open(training_inds_HANsFile, 'wb'))


generate_training_inds()
# use HANs
# with open(WESHCLASS_data_label_File, 'w') as my_file:
#     for line in labels:
#         print(line, file=my_file)


lines_tokenized_text = open(tokenized_text).readlines()
lines_per_doc = []
with open(WESHCLASS_data_File, 'w') as my_file:
    for i in indHANs2ind_superspans.values():
        line_per_doc = lines_tokenized_text[2 * i].strip() + ' ' + lines_tokenized_text[2 * i + 1].strip()
        lines_per_doc.append(line_per_doc)
        print(line_per_doc, file=my_file)


def re_generate_dataless():
    labels_per_doc = [l.strip() for l in open(superspan_HANs_labelsFile).readlines()]
    perm = list(np.random.permutation(len(labels_per_doc)))
    with open(dataless_data_permuted_File, 'w') as my_file, open(dataless_labels_permuted_File, 'w') as my_file_label:
        for i in perm:
            print(lines_per_doc[i], file=my_file)
            print(labels_per_doc[i], file=my_file_label)


re_generate_dataless()
