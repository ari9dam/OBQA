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
import collections

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


Input = collections.namedtuple("Input","idx premise hypothesis label")

class BertNLI:
    def __init__(self,output_dir,model="best_model.bin",topk=50,
                 bert_model="bert-large-cased",do_lower_case=False,train_batch_size=32,seed=42,
                 eval_batch_size=64,max_seq_length=128,num_labels=2,entail_label=1,grad_acc_steps=1,
                 num_of_epochs=5,learning_rate=1e-5,warmup_proportion=0.1,action="train"):
        
        print("Loading BERT NLI Model")
        self.name = "BertNLI"
        
        self.topk=topk
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        
        tokenizer = BertTokenizer.from_pretrained(bert_model,do_lower_case=do_lower_case)         
       
        self.device = device
        self.tokenizer=tokenizer
        self.max_seq_length = max_seq_length
        self.train_batch_size = int(train_batch_size / grad_acc_steps)
        self.eval_batch_size=eval_batch_size
        self.num_labels=num_labels
        self.entail_label = entail_label
        self.grad_acc_steps=grad_acc_steps
        self.gradient_accumulation_steps = self.grad_acc_steps
        self.num_of_epochs = num_of_epochs
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.n_gpu = n_gpu
        
        print("Action:",action)
        if action == "train":
            print("Params:")
            print("Device:",device)
            print("Max-Seq-Length:",max_seq_length)
            print("Train-Batch-Size:",train_batch_size)
            print("Num Labels:",num_labels)
            print("Gradient Accumulation Steps:",grad_acc_steps)
            print("Num of Train Epochs:",num_of_epochs)
            print("Output Dir:",output_dir)
            print("Learning Rate:",learning_rate)
            print("Warmup Proportion:",warmup_proportion)
            print("Number of Gpus:",n_gpu)


            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if n_gpu > 0:
                torch.cuda.manual_seed_all(seed)

            os.makedirs(self.output_dir, exist_ok=True)  
            model = BertForSequenceClassification.from_pretrained(bert_model,
                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
                  num_labels = num_labels)
            model.to(device)
            model = torch.nn.DataParallel(model) 
            self.model = model
            self.best_metric = None
        if action == "predict":
            output_model_file = os.path.join(output_dir,model)
            # Load a trained model that you have fine-tuned
            print("Loading Model",output_model_file)
            model_state_dict = torch.load(output_model_file)
            model = BertForSequenceClassification.from_pretrained(bert_model, state_dict=model_state_dict, num_labels=num_labels)
            model.to(device)
            model = torch.nn.DataParallel(model)
            self.model = model
        
    def create_examples(self,data):
        examples = []
        for idx,inp in tqdm(data.items(),desc="Generating Bert Examples:"):
            examples.append(InputExample(guid=inp.idx, text_a=inp.premise, text_b=inp.hypothesis, label=inp.label))
        return examples
    
    def get_features(self,eval_examples,fname):
#         tokenfile = "%s/%d-%s.tokens"%(self.output_dir,self.topk,fname) 
#         indexfile = "%s.index"%(tokenfile)
#         if os.path.isfile(tokenfile):
#                 pickle_in = open(tokenfile,"rb")
#                 features = pickle.load(pickle_in)
#                 pickle_in.close()
#                 pickle_in = open(indexfile,"rb")
#                 exindex = pickle.load(pickle_in)
#                 pickle_in.close()
#                 return features,exindex
        eval_features, evalindex = convert_examples_to_features(
            eval_examples, None, self.max_seq_length, self.tokenizer)
#         pickle_out = open(tokenfile,"wb") 
#         pickle.dump(eval_features, pickle_out)                                 
#         pickle_out.close()
#         pickle_out = open(indexfile,"wb") 
#         pickle.dump(evalindex, pickle_out)                                 
#         pickle_out.close()
        return eval_features,evalindex
    
    def get_dataloader(self,examples,fname):
        if examples is None:
            return None,None
        eval_features, evalindex = self.get_features(examples,fname)       
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_unique_ids = torch.tensor([f.uuid for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_unique_ids)
        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.train_batch_size)
        return eval_dataloader,evalindex
    
    def predict(self,data,no_cache=False):
        model = self.model
        device = self.device
        tokenizer = self.tokenizer
        n_gpu = self.n_gpu
        num_labels = self.num_labels
        val_examples = self.create_examples(data["val"])
        test_examples = self.create_examples(data["test"])
        test_dataloader,test_index = self.get_dataloader(test_examples,"test")
        val_dataloader,val_index = self.get_dataloader(val_examples,"val")
        valc,valinc=    self.predict_qa(val_dataloader,val_index,data["val"],model,"val",0)
        testc,testinc=    self.predict_qa(test_dataloader,test_index,data["test"],model,"test",0)    
        
        response = {
            "test": {
                "correct" : testc,
                "incorrect": testinc
            },
            "val" : {
                "correct" : valc,
                "incorrect" : valinc
            }
        }
        
        respfd = open(self.output_dir + "/eval_output.txt","w") 
        respfd.write(json.dumps(response))
        respfd.close()
        
        return response
    
    def train(self,data,no_cache=False):
        model =self.model
        device = self.device
        tokenizer=self.tokenizer
        learning_rate = self.learning_rate
        warmup_proportion = self.warmup_proportion
        num_labels = self.num_labels
        n_gpu = self.n_gpu
        
        # GET TRAIN SAMPLES - CACHE THE TOKENS
        # GET EVAL SAMPLES - CACHE THE TOKENS
        train_examples = self.create_examples(data["train"])
        val_examples = self.create_examples(data["val"],) if "val" in data else None
        test_examples_list = []
        for sample in data["test"]:
            test_examples = self.create_examples(sample)
            test_examples_list.append(test_examples)
        
        train_dataloader,train_index = self.get_dataloader(train_examples,"train")
        val_dataloader,val_index = self.get_dataloader(val_examples,"val")
        
        test_dataloader_list = []
        for index,examples in enumerate(test_examples_list):
            test_dataloader,test_index = self.get_dataloader(examples,"test"+str(index))
            test_dataloader_list.append((test_dataloader,test_index))
        
        num_train_steps = int(
            len(train_examples) / self.train_batch_size / self.gradient_accumulation_steps * self.num_of_epochs)
        
        # OPTIMIZERS MODEL INITIALIZATION
        
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=learning_rate,
                            warmup=warmup_proportion,
                            t_total=t_total)

        
        
        # TRAIN FOR EPOCHS and SAVE EACH MODEL as pytorch_model.bin.{epoch}
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0     
        ep = 0
        output_model_file = "dummy"
        loss_fct = CrossEntropyLoss()
        for _ in trange(int(self.num_of_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            ep += 1
            tq = tqdm(train_dataloader, desc="Iteration")
            acc = 0
            for step, batch in enumerate(tq):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids,unique_ids = batch
                logits = model(input_ids, segment_ids, input_mask)
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_accuracy = accuracy(logits, label_ids)
                acc += tmp_accuracy
                
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = self.learning_rate * warmup_linear(global_step/t_total, self.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
               
                tq.set_description("Loss:"+str(tr_loss/nb_tr_steps)+",Acc:"+str(acc/nb_tr_examples)) 
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(self.output_dir, "pytorch_model.bin." + str(ep))
            torch.save(model_to_save.state_dict(), output_model_file)
            
            # EVAL IN EACH EPOCH SAVE BEST MODEL as best_model.bin
            if val_dataloader:
                self.score_qa(val_dataloader,val_index,data["val"],model,"val",ep)
            if test_dataloader_list:
                for index,tup in enumerate(test_dataloader_list):
                    self.score_qa(tup[0],tup[1],data["test"][index],model,"test"+str(index),ep)     
            
            print("After Current-Epoch:",self.best_metric)
            
        return model,self.best_metric
        
        
    def predict_qa(self,eval_dataloader,eval_index,data,model,typ,ep):      
        device = self.device
        model.eval()
        nb_eval_steps, nb_eval_examples = 0, 0
        all_results = {}
        for input_ids, input_mask, segment_ids, label_ids,unique_ids in tqdm(eval_dataloader, desc="Predicting:"):
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
                idx = eval_index[q] 
                score = scores[self.entail_label]                                    
                qid = idx.split("__")[0]
                choice = idx[-1]
                if qid not in all_results:
                    all_results[qid]=[0,0,0,0]
                all_results[qid][int(choice)]=score
        
        count = 0
        inacc = 0
        corrects={}
        incorrects={}
        for qid,scores in all_results.items():
            maxidx = np.argsort(scores).tolist()[-1]
            if qid+"__ch_"+str(maxidx) in data:
                if data[qid+"__ch_"+str(maxidx)].label == 1:
                    count+=1
                    corrects[qid]=maxidx
                else:
                    incorrects[qid]=maxidx
            else:
                if data[qid].label == 1:
                    count+=1
                    corrects[qid]=maxidx
                else:
                    incorrects[qid]=maxidx
        
        acc = float(count)/float(len(all_results))
        logger.info("%s accuracy:%f"%(typ,acc))
        return corrects,incorrects
    
    def score_qa(self,eval_dataloader,eval_index,data,model,typ,ep):      
        device = self.device
        model.eval()
        nb_eval_steps, nb_eval_examples = 0, 0
        tq = tqdm(eval_dataloader, desc="Scoring:")
        all_results = {}
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
                idx = eval_index[q] 
                score = scores[self.entail_label]
                if "__" in idx:
                    qid = idx.split("__")[0]
                    choice = idx[-1]
                    if qid not in all_results:
                        all_results[qid]=[0,0,0,0]
                    all_results[qid][int(choice)]=score
                else:
                    qid = idx.split(":")[0]
                    choice = idx[-1]
                    if qid not in all_results:
                        all_results[qid]=[0,0,0,0]
                    all_results[qid][int(choice)]=score
                    
        
        count = 0
        inacc = 0
        for qid,scores in all_results.items():
            maxidx = np.argsort(scores).tolist()[-1]
            if qid+"__ch_"+str(maxidx) in data:
                if data[qid+"__ch_"+str(maxidx)].label == 1:
                    count+=1
                elif inacc < 5:
                    print(qid,maxidx,scores)
                    inacc +=1
            else:
                if data[qid+":"+str(maxidx)].label == 1:
                    count+=1
                elif inacc < 5:
                    print(qid,maxidx,scores)
                    inacc +=1
        
        acc = float(count)/float(len(all_results))
        logger.info("%s accuracy:%f"%(typ,acc))
        
        if self.best_metric is None:
            self.best_metric = {}
            self.best_metric["best-acc"] = acc
        
        self.best_metric["%s-%d"%(typ,ep)]=acc
        
        if self.best_metric["best-acc"] <= acc and typ == "val":
                # Save Best Val Accuracy Model
                self.best_metric["best-acc"] = acc
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(self.output_dir, "best_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
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

tokenmap = {}
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    #label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    exindex = {}
    passagelens = []
    
    sum_of_labels = 0
    
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
            
            passagelens.append(len(tokens_a)+len(tokens_b)+3)
            
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
        label_id = example.label
        
        sum_of_labels+=label_id
        
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
        
    
    print("Passage Token Lengths Distribution", max(passagelens), np.percentile(passagelens,50), np.percentile(passagelens,90), np.percentile(passagelens,95), np.percentile(passagelens,99))
    assert sum_of_labels > 0
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

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x