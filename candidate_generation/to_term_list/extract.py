import argparse
from rake_nltk import Rake
from collections import defaultdict
from summa.keywords import keywords

r = Rake()

parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
parser.add_argument('arg2', nargs='?', default="textrank", help="1st Positional arg")
args = parser.parse_args()
dataset = '/disk/home/klee/data/{}_merged_tokenized'.format(args.arg1)


def get_keyword_list(file, method='rake'):
    def get_termscoreList_from_text(extractedText):
        if method == 'textrank':
            import ipdb; ipdb.set_trace()
            return keywords(extractedText, scores=False)
        r.extract_keywords_from_text(extractedText)
        score_term_List = r.get_ranked_phrases_with_scores()
        return [(t[1], t[0]) for t in score_term_List]

    numLineReadEachExtraction = 10000
    if method == 'textrank':
        numLineReadEachExtraction = 1000
    f = open(file, "r")
    lineno = 0
    line = f.readline()
    lineno += 1
    termDict = defaultdict(int)
    lines = []

    while line:
        for i in range(0, numLineReadEachExtraction):
            if (line):
                lines.append(line)
                line = f.readline()
                lineno += 1
                print(lineno)
            else:
                break
        termScoreList = get_termscoreList_from_text(' '.join(lines))
        print('at lineno:{}'.format(lineno))
        for termTuple in termScoreList:
            term, score = termTuple
            termDict[term] = termDict[term] + score * len(lines) / numLineReadEachExtraction
            # if term in termDict:
            #     if (score > termDict[term]):
            #         termDict[term] = score
            # else:
            #     termDict[term] = score
        lines.clear()
    f.close()

    termFile = open(file + "_" + method + "_term.txt", "w")
    for term, score in sorted(termDict.items(), key=lambda x: x[1], reverse=True):
        if len(term.split(' ')) > 5:
            continue
        termFile.write(term)
        termFile.write("\t")
        termFile.write(str(score))
        termFile.write("\n")
    termFile.close()

get_keyword_list(dataset, args.arg2)