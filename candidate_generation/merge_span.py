# coding: utf-8
from collections import defaultdict, Counter
import spacy
import json
import os
from operator import itemgetter
from pprint import pprint
import sys
from tqdm import tqdm
import argparse
from m.util.common import make_parentdir, get_line_count, getLogger
import ipdb
import pickle


try:
    unicode
except Exception as e:
    # python 3
    unicode = str

log = getLogger(os.path.basename(__file__), use_stream=False)

parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
args = parser.parse_args()
tokenized_text = '/disk/home/klee/data/{}_merged_tokenized'.format(args.arg1)
outFile = tokenized_text + '_superspan_sequence.json'
source2freq_counterFile = tokenized_text + '_source2freq_counter.bin'


# intermediate
textFile = tokenized_text + ''
spacyFile = tokenized_text + '_spacy.json'
nltkFile = tokenized_text+'_nltk.json'
autoFile = tokenized_text + '_autophrase.json'
outListFile = tokenized_text + '_outListFile.json'


def read_span_json(inFile, num=None):
    if num:
        with open(inFile) as fin:
            data = []
            for i, line in enumerate(fin):
                data.append(json.loads(line))
                if i >= num:
                    return data

    return [json.loads(line) for line in open(inFile)]


def read_text(inFile, num=None):
    texts = []
    with open(inFile) as fin, open(textFile) as f_text:
        if num:
            for cnt, line in enumerate(f_text):
                texts.append(line)
                if cnt > num:
                    break
        else:
            for line in f_text:
                texts.append(line)
        return texts


# def get_line_count(inFile):
#     count = -1
#     for count, line in enumerate(open(inFile, 'r')):
#         pass
#     count += 1
#     return count


# # In[8]:


# def check_span_data(inFile_list):
#     for inFile in inFile_list:
#         print(inFile, 'line count', get_line_count(inFile))

# nltkData, spacyNPData, spacyEntityData, autoData
def merge_span_data(data_list, sourceNames=None):
    # check data length consistency
    length_list = [len(data) for data in data_list]
    print(length_list)
    assert len(set(length_list)) == 1

    length = length_list[0]
    new_data = []
    for i in range(length):
        new_d = []
        # todo: append sourcenames
        for data, source in zip(data_list, sourceNames):
            new_d.extend([dict(span, **{'source': source}) for span in data[i]])
        new_data.append(new_d)
    return new_data


def generate_superspan(d, tokens):
    cur_st = -1
    cur_ed = -1
    cur_superspan_ind = -1
    superspan = {}
    superspan_list = []
    for span in sorted(d, key=itemgetter('st', 'ed')):
        # span information
        st = span['st']
        ed = span['ed']
        text = span['text']
#       print('text', text)

        if st >= cur_ed:
            # meet a new superspan
            cur_st = st
            cur_ed = ed
            superspan = {'st': st,
                         'ed': ed, 'spans': []}
            superspan['spans'] = [span]
            superspan_list.append(superspan)
            cur_superspan_ind += 1
        else:
            # update super end pos
            cur_ed = max(ed, cur_ed)
            superspan_list[cur_superspan_ind]['ed'] = cur_ed
            if span not in superspan_list[cur_superspan_ind]['spans']:
                superspan_list[cur_superspan_ind]['spans'].append(span)
#             superspan_list[cur_superspan_ind]['text'] = ' '.join(tokens[cur_st:cur_ed])

    for i, superspan in enumerate(superspan_list):
        superspan_list[i]['text'] = ' '.join(
            tokens[superspan['st']:superspan['ed']])
    return superspan_list


# In[43]:


def remove_duplicate(x):
    # 0 is flag for plain text word
    occur = set()
    return_list = []
    for i in x:
        if i == 0 or i not in occur:
            return_list.append(i)
            occur.add(i)
    return return_list

# flag = [0, 1, 1, 0, 2, 3, 4, 5, 6, 7, 7,
#         8, 9, 9, 10, 11, 11, 12, 12, 12, 12, 0]
# remove_duplicate(flag)


# In[44]:

def validate_compress_superspan_sequence(superspan_sequence, tokens):
    # todo: compress
    for superspan in superspan_sequence:
        if superspan['tag'] == 'superspan':
            for span in superspan['spans']:
                try:
                    assert span['st'] >= superspan['st']
                    assert span['ed'] <= superspan['ed']
                    assert span['text'] == ' '.join(tokens[span['st']:span['ed']])
                except Exception as e:
                    import ipdb
                    ipdb.set_trace()
                    raise e
    return superspan_sequence


def generate_sequence(superspan_list, tokens):
    '''
        generate super span sequence
    '''
    flag = [0]*len(tokens)

    sequence = []
    for i, superspan in enumerate(superspan_list):
        st = superspan['st']
        ed = superspan['ed']
        for idx in range(st, ed):
            try:
                flag[idx] = i
                pass
            except Exception as e:
                import ipdb
                ipdb.set_trace()
                raise e

    flag = remove_duplicate(flag)

    for idx, v in enumerate(flag):
        if v == 0:
            sequence.append({'tag': 'plain', 'text': tokens[idx], 'st': idx, 'ed': idx+1})
        else:
            superspan_list[v].update({'tag': 'superspan'})
            sequence.append(superspan_list[v])
    return sequence


def remove_non_tail_span(superspan_list):
    try:
        new_superspan_list = []
        for superspan in superspan_list:
            new_span = []
            for span in superspan['spans']:
                if span['ed'] == superspan['super_ed']:
                    new_span.append(span)
            superspan['spans'] = new_span
            new_superspan_list.append(superspan)
    except Exception as e:
        print(e)
        ipdb.set_trace()
    return new_superspan_list


_nlp = spacy.load('en_core_web_sm')


def filter_span_data_by_grammar(data, texts):
    new_data = []
    for doc_ind, (line, spans) in tqdm(enumerate(zip(texts, data)), total=len(texts)):
        if not spans:
            new_data.append(spans)
            continue
        text = unicode(line.strip())
        tokens = text.split(' ')
        try:
            doc = _nlp(text, parse=False, tag=True, entity=False)
            pos_list = [token.pos_ for token in doc]
            assert len(tokens) == len(pos_list)
            new_spans = []
            for span in spans:
                if pos_list[span['ed']-1] not in ['NOUN', 'PROPN']:
                    log.info('doc_ind:{}, skipping {}'.format(doc_ind, span))
                    continue
                else:
                    ind = span['st']
                    while ind < span['ed']:
                        if pos_list[ind] not in ['NOUN', 'ADV', 'ADJ', 'PROPN']:
                            ind += 1
                        else:
                            break
                    if ind < span['ed']:
                        if span['st'] == ind:
                            new_span = span
                        else:
                            new_span = {'st': ind, 'ed': span['ed'], 'text': ' '.join(tokens[ind:span['ed']])}
                            log.info('modifying {} as {}'.format(span, new_span))
                        new_spans.append(new_span)
            new_data.append(new_spans)
        except Exception as e:
            log.error(e)
            new_data.append(spans)
            pass
    return new_data


def main():
    print('='*50 + 'merge_span_data')
    nltkData = read_span_json(nltkFile)
    spacyData = read_span_json(spacyFile)
    autoData = read_span_json(autoFile)
    texts = read_text(textFile)

    cutoff = min(len(nltkData), len(spacyData), len(autoData))
    nltkData = nltkData[:cutoff]
    spacyData = spacyData[:cutoff]
    autoData = autoData[:cutoff]
    texts = texts[:cutoff]

    # convert spacy data
    spacyNPData = []
    spacyEntityData = []
    for d in spacyData:
        spacyNPData.append(d['np'])
        spacyEntityData.append(d['entity'])

    new_data = merge_span_data(
        [nltkData, spacyNPData, spacyEntityData, autoData],
        ['nltk', 'spacy_np', 'spacy_entity', 'autophrase'])

    source2freq_counter = calc_phrase_freq(new_data)
    pickle.dump(source2freq_counter, open(source2freq_counterFile, 'wb'))

    print('='*50 + 'filter_span_data_by_grammar')
    new_data = filter_span_data_by_grammar(new_data, texts)

    print('='*50 + 'generate_sequence')
    with open(outListFile, 'w') as fout_list, open(outFile, 'w') as fout:
        lineCount = get_line_count(textFile)
        for doc_ind in tqdm(range(lineCount)):
            text = texts[doc_ind].strip()
            if text:
                tokens = text.split(' ')
            else:
                tokens = []

            superspan_list = generate_superspan(new_data[doc_ind], tokens)

            fout_list.write(json.dumps(superspan_list))
            fout_list.write('\n')

            superspan_sequence = generate_sequence(superspan_list, tokens)

            superspan_sequence = validate_compress_superspan_sequence(superspan_sequence, tokens)

            fout.write(json.dumps(superspan_sequence))
            fout.write('\n')


def calc_phrase_freq(data, islower=True):
    try:
        source2freq_counter = defaultdict(Counter)
        for index, spans in tqdm(enumerate(data), total=len(data)):
            for span in spans:
                if islower:
                    token = span['text'].lower()
                else:
                    token = span['text']
                source2freq_counter[span['source']][token] += 1
        return source2freq_counter
    except Exception as e:
        print(e)
        ipdb.set_trace()


if __name__ == '__main__':
    main()
