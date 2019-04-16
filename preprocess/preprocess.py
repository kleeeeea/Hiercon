from spacy.en import English
import spacy
from m.util.common import removeNonASCII, condenseSpace
import os
from m.util.file import batchClean_recur_merge
import json

import argparse
parser = argparse.ArgumentParser("An argparser.")
parser.add_argument('dataset', nargs='?', default="pubmed_descriptor", help="1st Positional arg")
args = parser.parse_args()
json_hiercon_bylabel_output_directory = '/disk/home/klee/data/{}'.format(args.dataset)
merged_output_file = json_hiercon_bylabel_output_directory + '_merged'
merged_output_file_tokenized = merged_output_file + '_tokenized'
MAXLINE = 10000

dataset = args.dataset
xmlfile = '/disk/home/klee/data/raw/desc2018'
jsonfile = xmlfile + '.json'
jsonfile_hiercon = xmlfile + '_hiercon.json'


def merge(json_hiercon_bylabel_output_directory, merged_output_file):

    def write_output_and_label(inputfile, f_out, f_out_label, relative_path):
        for doc_ind, l in enumerate(open(inputfile)):
            try:
                if doc_ind >= MAXLINE:
                    break
                obj = json.loads(l)

                if not obj.get('title') or not obj.get('abstract'):
                    continue

                print(removeNonASCII(obj['title']).replace('\n', ' ').replace('\r', ' '), file=f_out)
                print(removeNonASCII(obj['abstract']) .replace('\n', ' ').replace('\r', ' '), file=f_out)
                if args.dataset == 'pubmed_descriptor':
                    print(l.strip(), file=f_out_label)
                    continue
                print(os.path.basename(inputfile).rsplit('.', 1)[0], file=f_out_label)
            except Exception as e:
                raise e

    batchClean_recur_merge(json_hiercon_bylabel_output_directory, merged_output_file, write_output_and_label)




def tokenize(merged_output_file, merged_output_file_tokenized):
    nlp = spacy.load('en_core_web_sm')

    def tokenize_line(l):
        doc = nlp(l, parse=False, tag=False, entity=False)
        return condenseSpace(' '.join([str(t).replace(' ', '') for t in list(doc)]).strip())

    # tokenizer = English().Defaults.create_tokenizer(nlp)

    with open(merged_output_file_tokenized, 'w') as f:
        for l in open(merged_output_file):
            print(tokenize_line(l), file=f)


if __name__ == '__main__':
    merge(json_hiercon_bylabel_output_directory, merged_output_file)
    if dataset != 'pubmed_descriptor':
        tokenize(merged_output_file, merged_output_file_tokenized)
