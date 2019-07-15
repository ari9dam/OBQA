from models.spacy_based_ir import SpacyIR
from models.bert_sts import BertSTSIR
from models.bert_nli import BertNLIIR
from models.bert_cnn import BertCNNIR


from tqdm import tqdm
import argparse
import pickle
import numpy as np

def choose_model(topk=50):
    irmodel = SpacyIR(topk=50)
    return irmodel

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
   


#     irmodel.predict(data,outfile,tokenfile)
#     return
        
datasets = ["hyp-ques-test.tsv","hyp-ques-train.tsv","hyp-ques-val.tsv"]
datasets = ["hyp-ques-test.tsv","hyp-ques-val.tsv"]
    
# irmodel = choose_model(topk=50)
# data = read_data_to_score("openbook.txt")
# pred_data(data,irmodel,"../data/ranked/scapy-openbook.json")

# irmodel = BertSTSIR(topk=50,output_dir="/scratch/pbanerj6/stsb_output",model="pytorch_model.bin.3",eval_batch_size=256)
# data = read_data_to_score("openbook.txt",is_fact_fact=True)
# irmodel.predict(data,"../data/ranked/sts-factfact-orig.json","/scratch/pbanerj6/factfact.tokens")

# irmodel = BertSTSIR(topk=50,output_dir="/scratch/pbanerj6/stsb_output",model="pytorch_model.bin.3",eval_batch_size=1024)
# data = read_data_to_score("openbook.txt")
# irmodel.predict(data,"../data/ranked/sts-openbook.json","/scratch/pbanerj6/hypfacttokens/nli.tokens")

# irmodel = choose_model(topk=100)
# data = read_data_to_score("omcs.txt",datasets=datasets)
# irmodel.predict(data,"../data/ranked/scapy-omcs.json")


# irmodel = BertSTSIR(topk=50,output_dir="/scratch/pbanerj6/stsb_output",model="pytorch_model.bin.4",eval_batch_size=1024)
# data = read_data_to_score("openbook.txt",datasets=datasets)
# irmodel.predict(data,"../data/ranked/sts-factfact-orig.json","/scratch/pbanerj6/hyptestvaltokens/sts.tokens")

# irmodel = BertCNNIR(topk=50,output_dir="/scratch/pbanerj6/sml-class-bert-large-cnn-full",model="best_model.bin",eval_batch_size=1024)
# data = read_data_to_score("openbook.txt",datasets=datasets)
# irmodel.predict(data,"../data/ranked/cnn-openbook.json","/scratch/pbanerj6/cnntokens/nli.tokens")

irmodel = BertNLIIR(topk=50,output_dir="/scratch/pbanerj6/sml-class-bert-large-v2-5e6-full",model="best_model.bin",eval_batch_size=1024)
data = read_data_to_score("openbook.txt",datasets=datasets)
irmodel.predict(data,"../data/ranked/simplebert-openbook.json","/scratch/pbanerj6/cnntokens/nli.tokens")

# model_path = "/scratch/pbanerj6/qnli_orig_output/"
# model = "pytorch_model.bin.4"
# outfile = "../data/ranked/qnli-openbook.json"
# irmodel = BertNLIIR(topk=50,output_dir=model_path,model=model,eval_batch_size=2048)
# data = read_data_to_score("openbook.txt",datasets=datasets)
# irmodel.predict(data,outfile,"/scratch/pbanerj6/hyptestvaltokens/sts.tokens")