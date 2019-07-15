from models.spacy_based_ir import SpacyIR
from models.bert_sts import BertSTSIR
from models.bert_nli import BertNLIIR


from tqdm import tqdm
import argparse
import pickle
import numpy as np
import json

def read_data_to_score(factfile,is_fact_fact=False,datasets=None):
    data = {}
    base_dir = "../data/hypothesis/"
    if not is_fact_fact:
        fnames = datasets
    else:
        base_dir = "../data/knowledge/"
        fnames = ["openbook.txt"]
    
    facts = []
    factlines = open("../data/knowledge/"+factfile,"r").readlines()
    for fact in tqdm(factlines,desc="Processing Facts:"):
        fact=fact.strip().replace('"',"")
        facts.append(fact)
    
    for fname in fnames:
        lines = open(base_dir+fname,"r").readlines()   
        for index,line in tqdm(enumerate(lines),desc="Reading From "+fname+" :"):
            if not is_fact_fact:
                line = line.strip().split("\t")
                idx = line[0]
                choices = line[2:6]
                assert len(choices) == 4
                for index,choice in enumerate(choices):
                    nidx=idx+"__ch_"+str(index)
                    data[nidx]=choice
            else:
                line = line.strip().replace('"',"")
                data[str(index)]=line
  
    return {"data":data,"facts":facts}
   
def read_preranked_file(fname):
    preranked = {}
    lines = open("../data/ranked/"+fname,"r").readlines()
    for line in tqdm(lines,desc="Reading Pretrained"):
        line = line.strip()
        row = json.loads(line)
        preranked[row["id"]]=row
    return preranked
        
#datasets = ["hyp-ques-test.tsv","hyp-ques-train.tsv","hyp-ques-val.tsv"]
datasets = ["hyp-ques-test.tsv","hyp-ques-val.tsv"]
    
factfile = "omcs.txt"
prerankedfiles = ["tfidf-omcs.json","scapy-omcs.json"]

for rankedfile in prerankedfiles:
    for modeldir in tqdm(["/scratch/pbanerj6/stsir4_output/","/scratch/pbanerj6/stsb_output"],desc="Scoring :"+rankedfile):
        print("Running :",rankedfile,modeldir)
        irmodel = BertSTSIR(topk=50,output_dir=modeldir,model="pytorch_model.bin.4",eval_batch_size=1024)
        data = read_data_to_score(factfile,datasets=datasets)
        preranked = read_preranked_file(rankedfile)
        outputname = "sts.json" if "stsb" in modeldir else "trained.json"
        prefix = rankedfile.split(".")[0]
        irmodel.predict(data,"../data/ranked/"+prefix+"-"+outputname,"/scratch/pbanerj6/hyptestvaltokens/"+prefix+"-"+outputname+".tokens",preranked=preranked)


# irmodel = BertSTSIR(topk=50,output_dir="/scratch/pbanerj6/stsb_output",model="pytorch_model.bin.4",eval_batch_size=1024)
# data = read_data_to_score("openbook.txt",datasets=datasets)
# irmodel.predict(data,"../data/ranked/sts-factfact-orig.json","/scratch/pbanerj6/hyptestvaltokens/sts.tokens")

# model_path = "/scratch/pbanerj6/qnli_orig_output/"
# model = "pytorch_model.bin.4"
# outfile = "../data/ranked/qnli-openbook.json"
# irmodel = BertNLIIR(topk=50,output_dir=model_path,model=model,eval_batch_size=2048)
# data = read_data_to_score("openbook.txt",datasets=datasets)
# irmodel.predict(data,outfile,"/scratch/pbanerj6/hyptestvaltokens/sts.tokens")