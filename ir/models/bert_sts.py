from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pickle
import json

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class BertSTSIR:
    def __init__(self,output_dir,model,topk=50,
                 bert_model="bert-large-cased",do_lower_case=False,
                 eval_batch_size=64,max_seq_length=128,num_labels=1):
        print("Loading BERT STSIR Model")
        self.name = "BertSTSIR"
        self.topk=topk
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        
        tokenizer = BertTokenizer.from_pretrained(bert_model,do_lower_case=do_lower_case)
        output_model_file = os.path.join(output_dir,model)
        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(output_model_file)
        model = BertForSequenceClassification.from_pretrained(bert_model, state_dict=model_state_dict, num_labels=num_labels)
        model.half()
        model.to(device)
        model = torch.nn.DataParallel(model)        
        self.model = model
        self.device = device
        self.tokenizer=tokenizer
        self.max_seq_length = max_seq_length
        self.eval_batch_size=eval_batch_size
        self.num_labels=num_labels
        
    def create_eval_examples(self,hyps,facts,preranked=None):
        examples = []
        if preranked is None:
            for nidx,hyp in tqdm(hyps.items(),desc="Generating Bert Examples:"):
                for index,fact in enumerate(facts):
                    guid = nidx +":"+str(index)
                    examples.append(InputExample(guid=guid, text_a=hyp, text_b=fact, label=0))
        else:
            assert type(preranked) == dict
            for nidx,hyp in tqdm(hyps.items(),desc="Generating Bert Examples:"):
                factlist = preranked[nidx]["ext_fact_global_ids"]
                for index,tup in enumerate(factlist):
                    guid = nidx +":"+str(tup[0])
                    examples.append(InputExample(guid=guid, text_a=hyp, text_b=facts[tup[0]], label=0))
        return examples   
    
    def get_features(self,eval_examples,max_seq_length,tokenizer,tokenfile,chunk_id,no_cache=True):
        tokenfile = tokenfile+"."+str(chunk_id)
        indexfile = tokenfile+".index"
        if not no_cache:
            if os.path.isfile(tokenfile) :
                    pickle_in = open(tokenfile,"rb")
                    features = pickle.load(pickle_in)
                    pickle_in.close()
                    pickle_in = open(indexfile,"rb")
                    exindex = pickle.load(pickle_in)
                    pickle_in.close()
                    return features,exindex
        eval_features, evalindex = convert_examples_to_features(
            eval_examples, None, max_seq_length, tokenizer)
        pickle_out = open(tokenfile,"wb") 
        pickle.dump(eval_features, pickle_out)                                 
        pickle_out.close()
        pickle_out = open(indexfile,"wb") 
        pickle.dump(evalindex, pickle_out)                                 
        pickle_out.close()
        return eval_features,evalindex
    
    def predict(self,data,outfile,tokenfile,preranked=None):
        hyps = data["data"]
        facts = data["facts"]
        
        model =self.model
        device = self.device
        tokenizer=self.tokenizer
        max_seq_length=self.max_seq_length
        eval_batch_size=self.eval_batch_size
        num_labels=self.num_labels
        
        eval_examples = self.create_eval_examples(hyps,facts,preranked)
        
        all_results = {} 
        chunk_id = 0
        output = []
        for chunk in chunks(eval_examples,100000): 
            eval_features, evalindex = self.get_features(chunk,max_seq_length,tokenizer,tokenfile,chunk_id,no_cache=True)
            chunk_id+=1
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
            all_unique_ids = torch.tensor([f.uuid for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_unique_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            tq = tqdm(eval_dataloader, desc="Scoring:")
            for input_ids, input_mask, segment_ids, label_ids,unique_ids in tq:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                unique_ids = unique_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask)
                logits = logits.detach().cpu().tolist()                             
                label_ids = label_ids.detach().cpu().tolist()                       
                unique_ids = unique_ids.detach().cpu().tolist()

                nb_eval_examples += input_ids.size(0)                               
                nb_eval_steps += 1                                                  

                for q,scores in zip(unique_ids,logits):                             
                    idx = evalindex[q] 
                    nidx = idx.split(":")[0]
                    fid = idx.split(":")[1]
                    score = scores[0]                                    
                    if nidx not in all_results.keys():
                        all_results[nidx]=[0]*len(facts)
                    all_results[nidx][int(fid)]=score

        
        for nidx,scores in tqdm(all_results.items(),desc="Finding TopK:"):
            
            topkfacts = np.argsort(scores)[-self.topk:].tolist()
            out = {}
            out["id"]=nidx
            out["ext_fact_global_ids"]=[]
            for index in reversed(topkfacts):
                out["ext_fact_global_ids"].append([index,scores[index]])
            output.append(out)
            
        outfd = open(outfile,"w")
        for out in tqdm(output,desc="Writing to File:"):
            outfd.write(json.dumps(out)+"\n")
        outfd.close()
        
        return 

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
    
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,uuid, input_ids, input_mask, segment_ids, label_id):
        self.uuid = uuid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

tokenmap= {}
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    #label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    exindex = {}
    for (ex_index, example) in tqdm(enumerate(examples),desc="Tokenizing:"):
        if example.text_a not in tokenmap.keys():
            tokens_a = tokenizer.tokenize(example.text_a)
            tokenmap[example.text_a]=tokens_a
        else:
            tokens_a = tokenmap[example.text_a]

        tokens_b = None
        if example.text_b:           
            if example.text_b not in tokenmap.keys():
                tokens_b = tokenizer.tokenize(example.text_b)
                tokenmap[example.text_b]=tokens_b
            else:
                tokens_b = tokenmap[example.text_b]
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        #label_id = label_map[example.label]
        label_id = 0.0 #example.label
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (str(example.label), 0))
        
        exindex[ex_index]=example.guid
        features.append(
                InputFeatures(uuid = ex_index,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features,exindex


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def average(x):                                                                 
    assert len(x) > 0                                                           
    return float(sum(x)) / len(x) 

import math                                                                     
def pearson(x, y):                                                              
    xx = []                                                                     
    for val in x:                                                               
        xx.append(val[0])                                                       
    x = xx                                                                      
    assert len(x) == len(y)                                                     
    n = len(x)                                                                  
    assert n > 0                                                                
    avg_x = average(x)                                                          
    avg_y = average(y)                                                          
    diffprod = 0                                                                
    xdiff2 = 0                                                                  
    ydiff2 = 0                                                                  
    for idx in range(n):                                                        
        xdiff = x[idx] - avg_x                                                  
        ydiff = y[idx] - avg_y                                                  
        diffprod += xdiff * ydiff                                               
        xdiff2 += xdiff * xdiff                                                 
        ydiff2 += ydiff * ydiff                                                 
                                                                                
    return diffprod / math.sqrt(xdiff2 * ydiff2) 