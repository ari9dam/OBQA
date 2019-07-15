import spacy
import time
from tqdm import tqdm
import numpy as np
import json


class SpacyIR:
    def __init__(self,topk=50):
        print("Loading Spacy Large Model")
        self.name = "SpacyIR"
        self.model =  spacy.load('en_vectors_web_lg')
        self.topk=topk
    def score(self,p,h):
        doc1 = self.model(p)
        doc2 = self.model(h)
        return doc1.similarity(doc2)
    
    def predict(self,data,outfile,tokenfile=None):
        hyps = data["data"]
        facts = data["facts"]
        fdocs = []
        for fact in tqdm(facts,desc="Tokenizing Facts:"):
            fdocs.append(self.model(fact))
        hypdocs = []
        for nidx,hyp in tqdm(hyps.items(),desc="Tokenzing Docs:"):
            assert hyp != ""
            doc = self.model(hyp)
            hypdocs.append((nidx,doc))
            
        self.model = None
           
        print("Number of Facts:",len(fdocs))
        output = []                
        for nidx,doc in tqdm(hypdocs,desc="Predicting Scores:"):
            out = {}
            similarities = []
            for fdoc in fdocs:
                similarities.append(fdoc.similarity(doc))
            topkfacts = np.argsort(similarities)[-self.topk:].tolist()
            out["id"]=nidx
            out["ext_fact_global_ids"]=[]
            for index in reversed(topkfacts):
                out["ext_fact_global_ids"].append([index,similarities[index]])
            output.append(out)

            
        outfd = open(outfile,"w")
        for out in tqdm(output,desc="Writing to File:"):
            outfd.write(json.dumps(out)+"\n")
        outfd.close()
        return