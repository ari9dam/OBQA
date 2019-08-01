import json
import collections
import numpy as np
from tqdm import tqdm
import operator

Entry = collections.namedtuple("Entry","qid fact1 fact2 hyp1 hyp2 hyp3 hyp4 ans label")

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

def read_hyp_dataset(fname):
    fd = open(fname,"r").readlines()
    dataset = {}
    for line in tqdm(fd,desc="Reading Datasets:"):
        line = line.strip().split("\t")
        qid = line[0]
        passage = line[1].split(" . ")
        choice = line[2:6]
        label = line[6]
        ans = choice[int(label)]
        fact1 = passage[0].strip()
        fact2 = passage[1].strip()
        entry = Entry(qid=qid,fact1=fact1,fact2=fact2,hyp1=choice[0],hyp2=choice[1],hyp3=choice[2],hyp4=choice[3],ans=ans,label=int(label))
        dataset[qid]=entry
    return dataset

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
#         if qid == "8-343":
#             print(merged[qid])
    return merged

def score_ranked_fact1(ranked,datasets,knowledgemap,knowledge,is_merged=False):
    topklist = [1,3,5,10,20,50]
    choices =  ["__ch_0","__ch_1","__ch_2","__ch_3"]
    for dataset in datasets:
        counts=[0,0,0,0,0,0]
        counts_ans=[0,0,0,0,0,0]
        pr=5
        print("For Dataset:")
        for index,topk in enumerate(topklist):
            for qid,entry in dataset.items():
                fact1 = entry.fact1
                label = entry.label
                fidx = knowledgemap[fact1.strip().lower()]
                if pr<5:
                    print(qid,fact1,fidx)
                found = False
                found_ans = False
                for choice in choices:
                    if not is_merged:
                        idx = qid+choice
                    else:
                        idx=qid
                    ranked_list=ranked[idx]
                    processed = [tup[0] for tup in ranked_list]
                    if fidx in processed[0:topk]:
                        found=True
                    if choice[-1]==str(label) and fidx in processed[0:topk]:
                        found_ans=True
                    if pr<5:
                        for f in processed[0:1]:
                            prob=0
                            for tup in ranked_list:
                                if tup[0] == f:
                                    prob = tup[1]
                            print(qid,"\t","\t",choice,knowledge[f],f,prob)
                pr+=1    
                if found:
                    counts[index]+=1
                if found_ans:
                    counts_ans[index]+=1
                        
            

                        
        print("Counts@\t1,3,5,10,20,50\n")
        print("\t",counts)
        print("\t",counts_ans)
        

knowledgemap,knowledge = read_knowledge("../data/knowledge/openbook.txt")

test = read_hyp_dataset("../data/hypothesis/hyp-gold-test.tsv")
train = read_hyp_dataset("../data/hypothesis/hyp-gold-train.tsv")
val = read_hyp_dataset("../data/hypothesis/hyp-gold-val.tsv")

ranked_spacy = read_ranked("../data/ranked/scapy-openbook.json",50)
ranked_sts = read_ranked("../data/ranked/sts-openbook.json",50)
ranked_trained = read_ranked("../data/ranked/sts-trained-openbook.json",50)
ranked_tfidf = read_ranked("../data/ranked/tfidf-openbook.json",50)
ranked_qnli = read_ranked("../data/ranked/qnli-openbook.json",50)
ranked_simple = read_ranked("../data/ranked/simplebert-openbook.json",50)
ranked_cnn = read_ranked("../data/ranked/cnn-openbook.json",50)

print("Scoring Unmerged")

# for ranked in [ranked_spacy,ranked_sts,ranked_trained,ranked_tfidf,ranked_qnli,ranked_simple,ranked_cnn]:
for ranked,name in zip([ranked_tfidf,ranked_sts,ranked_trained,ranked_cnn,ranked_simple],["tfidf","sts","trained","cnn","simple"]):
    print("Model:",name)
    print("Val")
    score_ranked_fact1(ranked,[val],knowledgemap,knowledge,is_merged=False)
    print("Test")
    score_ranked_fact1(ranked,[test],knowledgemap,knowledge,is_merged=False)
