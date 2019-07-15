#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from copy import deepcopy
import codecs
from sacremoses import MosesDetokenizer
from conllu import parse
from rule import Question, AnswerSpan
import pattern
import argparse


def qa2d(idx):
    q = Question(deepcopy(examples[idx].tokens))
    if not q.isvalid:
        print("Question {} is not valid.".format(idx))
        return ''
    a = AnswerSpan(deepcopy(examples[str(idx)+'_answer'].tokens))
    if not a.isvalid:
        print("Answer span {} is not valid.".format(idx))
        return ''
    q.insert_answer_default(a)
    return detokenizer.detokenize(q.format_declr(), return_str=True)


def print_sentence(idx):
    return detokenizer.detokenize([examples[idx].tokens[i]['form'] for i in range(len(examples[idx].tokens))], return_str=True)


def writeToFile(outputfile,examples):
    with open(outputfile,"w") as f:
        total = int(len(examples.keys())/2)
        print("Transforming {} examples.".format(total))
        for i in range(total):
            out = qa2d(i)
            #print(print_sentence(i))
            f.write(print_sentence(i)+"\n")
            if out != '':
                #print(out)
                f.write(out+"\n")
            #print('----------') 
            f.write("\n")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile",help="input parsed and tagged file")
    parser.add_argument("--outputfile",help="output q/a file")
    args = parser.parse_args()

    detokenizer = MosesDetokenizer()

    #inputfile = 'examples.conllu'
    #inputfile = 'parser_output.txt'

    inputfile = args.inputfile
    outputfile = args.outputfile

    # ## Parse conllu file

    print('Parsing conllu file...')
    with codecs.open(inputfile, 'r', encoding='utf-8') as f:
        conllu_file = parse(f.read())

    # Creating dict
    ids = range(int(len(conllu_file)/2))
    examples = {}
    count = 0
    for i, s in enumerate(conllu_file):
        if i % 2 == 0:
            examples[ids[count]] = s
        else:
            examples[str(ids[count])+'_answer'] = s
            count +=1
    
    writeToFile(outputfile,examples)