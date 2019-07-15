from models.bert_qa import BertQA

from tqdm import tqdm
import argparse
import pickle
import numpy as np
import os
import collections
import json
import operator

Input = collections.namedtuple("Input","idx passage a b c d label")

def read_ranked(fname,topk):
    fd = open(fname,"r").readlines()
    ranked={}
    for line in tqdm(fd,desc="Ranking "+fname+" :"):
        line = line.strip()
        out = json.loads(line)
        ranked[out["id"]]=out["ext_fact_global_ids"][0:topk]
    return ranked

def read_knowledge(fname):
    lines = open(fname,"r").readlines()
    knowledgemap = {}
    knowledge=[]
    for index,fact in tqdm(enumerate(lines),desc="Reading Knowledge:"):
        f=fact.strip().replace('"',"").lower()
        knowledgemap[f]=index
        knowledge.append(f)
    return knowledgemap,knowledge

# Returns itself if topk and train
def get_train_facts(fact1,topk,ranked,knowlegde,knowledgemap):
    if topk == 1:
        return [fact1]
    fset = []
    similarfacts = ranked[str(knowledgemap[fact1.lower()])]
    #print(similarfacts)
    for tup in similarfacts:
        f = knowledge[tup[0]]
        fset.append(f)
    if fact1.lower() in fset:
        return fset
    else :
        fset = fset[0:len(fset)-1]
        fset.append(fact1)
        assert len(fset) == topk
    return fset
    

def read_data_to_train(topk,ranked,knowledge,knowledgemap,use_gold_f2=False):
    tfile = "../data/generated/qa-"+str(topk)+str(use_gold_f2)+"-train.pickle"
    if os.path.isfile(tfile):
        pickle_in = open(tfile,"rb")
        features = pickle.load(pickle_in)
        pickle_in.close()
        if len(features)!=0:
            return features    
    data = {}
    gold = open("../data/hypothesis/hyp-gold-train.tsv","r").readlines()
    idx=0
    for line in tqdm(gold,desc="Preparing Train Dataset:"):
        line = line.strip().split("\t")
        qid = line[0]
        passage = line[1].split(" . ")
        choices = line[2:6]
        label = line[6]
        ans = choices[int(label)]
        fact1 = passage[0].strip()
        fact2 = passage[1].strip()
        if use_gold_f2:
            premise = line[1]
            data[idx]=Input(idx=idx,passage=premise,a=choices[0],b=choices[1],c=choices[2],d=choices[3],label=int(label))
        else:
            simfacts = get_train_facts(fact1,topk,ranked,knowledge,knowledgemap)
            premise = simfacts[0] 
            for fact in simfacts[1:]:
                premise = premise + " . " + fact
            data[idx]=Input(idx=idx,passage=premise,a=choices[0],b=choices[1],c=choices[2],d=choices[3],label=int(label))
        ##To make balanced dataset, repeating right answer 2 more times.
        idx+=1
    pickle_out = open(tfile,"wb+") 
    pickle.dump(data, pickle_out)                                 
    pickle_out.close()
    return data
        
def read_data_to_test(fname,topk,ranked,knowledge,knowledgemap,is_merged=False,use_gold_f2=False,cache=False):
    vfile = "../data/generated/qa-"+str(topk)+"-"+fname+"-"+str(use_gold_f2)+".pickle" if not is_merged else "../data/generated/qa-merged-"+str(topk)+"-"+fname+"-"+str(use_gold_f2)+".pickle"
    if os.path.isfile(vfile) and cache:
        pickle_in = open(vfile,"rb")
        features = pickle.load(pickle_in)
        pickle_in.close()
        if len(features)!=0:
            return features 
    data = {}
    val = open("../data/hypothesis/"+fname,"r").readlines()
    for line in tqdm(val,desc="Preparing Test Dataset:"):
        line = line.strip().split("\t")
        qid = line[0]
        passage = line[1]
        choices = line[2:6]
        label = line[6]
        ans = choices[int(label)]
        fact1 = passage[0].strip()
        fact2 = passage[1].strip()
        if use_gold_f2:
            premise = line[1]
            data[qid] = Input(idx=qid,passage=premise,a=choices[0],b=choices[1],c=choices[2],d=choices[3],label=int(label))
        else:
            for index,choice in enumerate(choices):
                qidx = qid if is_merged else qid+"__ch_"+str(index)
                simfacts = ranked[qidx][0:topk]
                qidx = qid+":"+str(index)
                premise = knowledge[simfacts[0][0]]
                for fidx in simfacts[1:]:
                    premise = premise + " . " + knowledge[fidx[0]]
                data[qidx]=Input(idx=qidx,passage=premise,a=choices[0],b=choices[1],c=choices[2],d=choices[3],label=int(label))
    
    pickle_out = open(vfile,"wb+") 
    pickle.dump(data, pickle_out)                                 
    pickle_out.close()
    return data

def gen_data_to_ir(fname,topk,ranked,knowledge,knowledgemap,is_merged=False): 
    val = open("../data/hypothesis/"+fname,"r").readlines()
    ofd = open("../data/merged/"+fname,"w+")
    for line in tqdm(val,desc="Preparing Train Dataset:"):
        line = line.strip().split("\t")
        qid = line[0]
        passage = line[1].split(" . ")
        choices = line[2:6]
        label = line[6]
        ans = choices[int(label)]
        fact1 = passage[0].strip()
        fact2 = passage[1].strip()
        simfacts = ranked[qid][0:topk]
        for findex,fidx in enumerate(simfacts):
            fact = knowledge[fidx[0]]
            score= fidx[1]
            for cindex,choice in enumerate(choices):
                nlabel = 1 if choice == ans else 0
                widx = qid+":"+str(cindex)+":"+str(findex)
                ofd.write("%s\t%s\t%s\t%d\t%f\n"%(widx,fact,choice,nlabel,score))
    ofd.close()
            
    return 

def merge_ranked(ranked):
    tmerged={}
    merged={}
    for qidx in tqdm(ranked.keys(),desc="Merging"):
        qid = qidx.split("__")[0]
        choice = qidx[-1]
        if qid not in tmerged:
            tmerged[qid]={}
        scores=ranked[qidx]
        for tup in scores:
            if tup[0] not in tmerged[qid]:
                tmerged[qid][tup[0]]=tup[1]
            else:
                tmerged[qid][tup[0]]+=tup[1]

        sorted_x = sorted(tmerged[qid].items(), key=operator.itemgetter(1))
        ranked_list=[]
        for tup in reversed(sorted_x):
            ranked_list.append(tup)
        merged[qid]=ranked_list
    return merged
    
def print_qa_inputs(data,typ):
    print("Type :",typ)
    keys = list(data.keys())[0:5]
    for i in keys:
        print(data[i])

        
parser = argparse.ArgumentParser()

parser.add_argument("--exp",
                        default=None,
                        type=str,
                        required=True,
                        help="Experiment name")
parser.add_argument("--method",
                        default="sum",
                        type=str,
                        required=True,
                        help="Score Method, sum/max")
parser.add_argument("--max_seq",
                        default=128,
                        type=int,
                        required=False,
                        help="Sequence Length")
parser.add_argument("--topk", default=1, type=int, required=True,
                        help="TopK Facts to fetch")
parser.add_argument("--merged",
                        action='store_true',
                        help="Whether not to merge facts")
parser.add_argument("--gold",
                        action='store_true',
                        help="Use Gold Facts")
parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help="random seed for initialization")

args = parser.parse_args()
    
exp = args.exp
topk = args.topk
max_seq = args.max_seq
seed = args.seed
epochs = args.epochs
method = args.method

is_merged = args.merged
use_gold_f2 = args.gold


output_dir = "/scratch/pbanerj6/bertqa-"+exp+"-"+str(topk)+"-"+str(is_merged)+"-"+str(max_seq)+"/"

ranked_factfact = read_ranked("../data/ranked/sts-factfact.json",topk=topk)
ranked_trained = read_ranked("../data/ranked/sts-trained-openbook.json",topk=topk)
# ranked_trained = read_ranked("../data/ranked/sts-openbook.json",topk=topk)
# ranked_trained = read_ranked("../data/ranked/tfidf-openbook.json",topk=topk)
knowledgemap,knowledge = read_knowledge("../data/knowledge/openbook.txt")


# ranked_trained = read_ranked("../data/ranked/tfidf-omcs-trained.json",topk=topk)
# knowledgemap,knowledge = read_knowledge("../data/knowledge/omcs.txt")

if is_merged:
    ranked_trained = merge_ranked(ranked_trained)

# gen_data_to_ir("hyp-gold-val.tsv",topk,ranked_trained,knowledge,knowledgemap,is_merged=is_merged)
# gen_data_to_ir("hyp-gold-test.tsv",topk,ranked_trained,knowledge,knowledgemap,is_merged=is_merged)
    
traindata = read_data_to_train(topk,ranked_factfact,knowledge,knowledgemap,use_gold_f2=use_gold_f2)
valdata = read_data_to_test("hyp-gold-val.tsv",topk,ranked_trained,knowledge,knowledgemap,is_merged=is_merged,use_gold_f2=use_gold_f2)
testdata = read_data_to_test("hyp-gold-test.tsv",topk,ranked_trained,knowledge,knowledgemap,is_merged=is_merged,use_gold_f2=use_gold_f2)
# valdata = read_data_to_test("hyp-gold-val-quant.tsv",topk,ranked_trained,knowledge,knowledgemap,is_merged=is_merged)
# testdata = read_data_to_test("hyp-gold-test-quant.tsv",topk,ranked_trained,knowledge,knowledgemap,is_merged=is_merged)

print_qa_inputs(traindata,"Train")
print_qa_inputs(valdata,"Val")
print_qa_inputs(testdata,"Test")

model = BertQA( output_dir=output_dir,topk=topk,
                 bert_model="bert-large-cased",do_lower_case=False,train_batch_size=32,seed=seed,
                 eval_batch_size=32,max_seq_length=max_seq,num_labels=4,grad_acc_steps=2,
                 num_of_epochs=epochs,action="train")

data = { "train":traindata,"val":valdata, "test":[testdata]}

best = model.train(data,method)

# model = BertQA( output_dir=output_dir,topk=topk,
#                  bert_model="bert-large-cased",do_lower_case=False,train_batch_size=64,seed=seed,
#                  eval_batch_size=64,max_seq_length=max_seq,num_labels=4,grad_acc_steps=2,
#                  num_of_epochs=epochs,action="predict")
# model.predict(data)