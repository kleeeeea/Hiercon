import os


import argparse
import sys
import json
from tqdm import tqdm
import ipdb
import re
from Hiercon.candidate_generation.to_json.nltk_extract import validate_nps
import subprocess
from m.util.common import get_line_count, condenseSpace


AUTOPHRASE_PATH = '/disk/home/klee/workspace_remote/Hiercon/candidate_generation/AutoPhrase'
parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
args = parser.parse_args()
tokenized_text = '/disk/home/klee/data/{}_merged_tokenized'.format(args.arg1)


def model2segmented_text(MODEL):
    return os.path.join(AUTOPHRASE_PATH, MODEL, "segmentation.txt")


MODEL = args.arg1
tokenized_text_autophrase = tokenized_text + '_autophrase.json'


def train_autophrase(text_to_seg, model=MODEL):
    os.environ['RAW_TRAIN'] = text_to_seg
    os.environ['MODEL'] = model
    mycwd = os.getcwd()
    os.chdir(AUTOPHRASE_PATH)
    proc = subprocess.Popen('bash auto_phrase.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line)
    os.chdir(mycwd)


def segment(text_to_seg, model=MODEL):
    os.environ['TEXT_TO_SEG'] = text_to_seg
    os.environ['MODEL'] = model
    mycwd = os.getcwd()
    os.chdir(AUTOPHRASE_PATH)
    proc = subprocess.Popen('bash phrasal_segmentation.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line)
    os.chdir(mycwd)
#


def removeMarker(text):
    # '<phrase>'
    return re.sub('</?phrase>', '', text)
# [\w ]+</phrase>

# def validate_nps(nps, tokens):
#     for np in nps:
#         st = np['st']
#         ed = np['ed']
#         token_span = tokens[st:ed]
#         np_span = np['text'].split(' ')
#         if token_span != np_span:
#             print('token span', token_span, 'np_span', np_span)
#             return False
#     return True


def writeToJson(inFile, outFile, originalFile):
    with open(inFile, 'r') as fin, open(outFile, 'w') as fout, open(originalFile, 'r') as fOriginal:
        # with open(inFile, 'r') as fin:
        total = get_line_count(inFile)

        cnt = 0
        data = []
        for i, (line, line_original) in tqdm(enumerate(zip(fin, fOriginal)), total=total):
            text = line.strip()

            tokens = text.split(' ')
            original_tokens = line_original.split()
            clean_tokens = condenseSpace(removeMarker(text)).split(' ')
            nps = []
            for idx, token in enumerate(tokens):
                if '<phrase>' in token:
                    if token.startswith('<phrase>'):
                        span = {'st': idx}
                    else:
                        span = {}
                elif '</phrase>' in token:
                    if token.endswith('</phrase>'):
                        try:
                            if span:
                                span['ed'] = idx + 1
                                span['text'] = ' '.join(clean_tokens[span['st']:span['ed']])
                                nps.append(span)
                            span = {}
                        except Exception as e:
                            ipdb.set_trace()
                            print(e)
                    else:
                        span = {}
            if nps:
                nps_v = validate_nps(nps, original_tokens)
                if nps_v != nps:
                    ipdb.set_trace()
                nps = nps_v
            fout.write(json.dumps(nps))
            fout.write('\n')

            # fout.write(json.dumps(nps))
            # fout.write('\n')


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    if MODEL == 'pubmed_cleaned':
        segment(tokenized_text, 'pubmed')
    else:
        print('train_autophrase' + '='*60)
        train_autophrase(tokenized_text, args.arg1)
        print('segment' + '='*60)
        segment(tokenized_text, args.arg1)
    segmented_text = model2segmented_text(MODEL)
    writeToJson(segmented_text, tokenized_text_autophrase, tokenized_text)

    # inFile = sys.argv[1]
    # outFile = sys.argv[2]
    # # inFile = "/scratch/home/hwzha/workspace/AutoPhrase/models/test/segmentation.txt"
    # # outFile = "/scratch/home/hwzha/workspace/auto/result/test/merged.txt_without_sentence_id.json"
