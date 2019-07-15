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
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from torch.nn import CrossEntropyLoss                                           
from scipy.stats import rankdata 
import math

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


Input = collections.namedtuple("Input","idx passage a b c d label")

class BertQA:
    def __init__(self,output_dir,model="best_model.bin",topk=1,
                 bert_model="bert-large-cased",do_lower_case=False,train_batch_size=32,seed=42,
                 eval_batch_size=64,max_seq_length=128,num_labels=4,grad_acc_steps=1,
                 num_of_epochs=5,learning_rate=1e-5,warmup_proportion=0.1,action="train",fp16=False,loss_scale=0):
        
        print("Loading BERT QA Model")
        self.name = "BertQA"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        
        tokenizer = BertTokenizer.from_pretrained(bert_model,do_lower_case=do_lower_case)         
       
        self.topk=topk
        self.device = device
        self.tokenizer=tokenizer
        self.max_seq_length = max_seq_length
        self.train_batch_size = int(train_batch_size / grad_acc_steps)
        self.eval_batch_size=eval_batch_size
        self.num_labels=num_labels
        self.grad_acc_steps=grad_acc_steps
        self.gradient_accumulation_steps = self.grad_acc_steps
        self.num_of_epochs = num_of_epochs
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.n_gpu = n_gpu
        self.fp16=fp16
        self.loss_scale=loss_scale
        
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

            os.makedirs(self.output_dir, exist_ok=True)  
            model = BertForMultipleChoice.from_pretrained(bert_model,
                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
                  num_choices = num_labels)
            if self.fp16:
                print("Model: FP16")
                model.half()
            model.to(device)
            model = torch.nn.DataParallel(model) 
            self.model = model
            self.best_metric = None
            
        if action == "predict":
            output_model_file = os.path.join(output_dir,model)
            # Load a trained model that you have fine-tuned
            self.train_batch_size=80
            print("Loading Model",output_model_file)
            model_state_dict = torch.load(output_model_file)
            model = BertForMultipleChoice.from_pretrained(bert_model, state_dict=model_state_dict, num_choices=num_labels)
            model.to(device)
            model = torch.nn.DataParallel(model)
            self.model = model
        
    def create_examples(self,data):
        examples = []
        for idx,inp in tqdm(data.items(),desc="Generating Bert Examples:"):
            examples.append(InputExample(idx=inp.idx, context_sentence=inp.passage, ending_0=inp.a,ending_1=inp.b,ending_2=inp.c,ending_3=inp.d, label=inp.label))
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
            eval_examples, self.tokenizer, self.max_seq_length, self.tokenizer)
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
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        all_unique_ids = torch.tensor([f.example_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_unique_ids)
        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.train_batch_size)
        return eval_dataloader,evalindex
    
    def predict(self,data,no_cache=False,method="sum"):
        model = self.model
        device = self.device
        tokenizer = self.tokenizer
        n_gpu = self.n_gpu
        num_labels = self.num_labels
        val_examples = self.create_examples(data["val"])
        test_examples = self.create_examples(data["test"])
        test_dataloader,test_index = self.get_dataloader(test_examples,"test")
        val_dataloader,val_index = self.get_dataloader(val_examples,"val")
        acc,valincsum,valincmax,amax,asum=    self.predict_qa(val_dataloader,val_index,data["val"],model,"val",0,method)
        acc,testincsum,testincmax,amax,asum=    self.predict_qa(test_dataloader,test_index,data["test"],model,"test",0,method)    
        
        response = {
            "test": {
                "sum" : testincsum,
                "max": testincmax
            },
            "val" : {
                "correct" : valincsum,
                "incorrect" : valincmax
            }
        }
        
        respfd = open(self.output_dir + "/eval_output.txt","w") 
        respfd.write(json.dumps(response))
        respfd.close()
        
        return response
    
    def predict_single(self,data,no_cache=False,method="sum"):
        model = self.model
        device = self.device
        tokenizer = self.tokenizer
        n_gpu = self.n_gpu
        num_labels = self.num_labels
        test_examples = self.create_examples(data["test"])
        test_dataloader,test_index = self.get_dataloader(test_examples,"test")
        acc,testincsum,testincmax,amax,asum=    self.predict_qa(test_dataloader,test_index,data["test"],model,"test",0,method)    
        
        response = {
            "test": {
                "sum" : testincsum,
                "max": testincmax
            }
        }
        
        return response,amax,asum
    
    def predict_multi(self,data,epoch,no_cache=False,write_dir=None):
        model = self.model
        device = self.device
        tokenizer = self.tokenizer
        n_gpu = self.n_gpu
        num_labels = self.num_labels
        
        test_examples_list = []
        for sample in data["test"]:
            test_examples = self.create_examples(sample,) if "test" in data else None
            test_examples_list.append(test_examples)
        test_dataloader_list = []
        for index,examples in enumerate(test_examples_list):
            test_dataloader,test_index = self.get_dataloader(examples,"test"+str(index))
            test_dataloader_list.append((test_dataloader,test_index))
        
        response = {}
        incorrects = {}
        
        for index,tup in enumerate(test_dataloader_list):
            acc,inc_sum,inc_max,amax,asum=self.predict_qa(tup[0],tup[1],data["test"][index],model,"test"+str(index),0)    
            response["test"+str(index)]=acc
            incorrects["test"+str(index)+"sum"]=inc_sum
            incorrects["test"+str(index)+"max"]=inc_max
        
        if write_dir is None:
            write_dir = self.output_dir
        
        respfd = open(write_dir + "/"+str(epoch)+"_eval_output.txt","w+") 
        respfd.write(json.dumps(response))
        respfd.close()
        incofd = open(write_dir + "/"+str(epoch)+"_incorrect_scores.txt","w+")
        incofd.write(json.dumps(incorrects))
        incofd.close()
        return response
    
    def train(self,data,no_cache=False,method="sum"):
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
            test_examples = self.create_examples(sample,) if "test" in data else None
            test_examples_list.append(test_examples)
        
        train_dataloader,train_index = self.get_dataloader(train_examples,"train")
        
        test_dataloader_list = []
        for index,examples in enumerate(test_examples_list):
            test_dataloader,test_index = self.get_dataloader(examples,"test"+str(index))
            test_dataloader_list.append((test_dataloader,test_index))
        val_dataloader,val_index = self.get_dataloader(val_examples,"val")
        
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
#         optimizer = BertAdam(optimizer_grouped_parameters,
#                             lr=learning_rate,
#                             warmup=warmup_proportion,
#                             t_total=t_total)

        if self.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=self.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            print("Optimizer: FusedAdam")
            if self.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=self.learning_rate,
                                 warmup=self.warmup_proportion,
                                 t_total=num_train_steps)
  
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
                loss = loss_fct(logits, label_ids)                               
                logits = logits.detach().cpu().numpy()                          
                label_ids = label_ids.to('cpu').numpy() 
                tmp_accuracy = accuracy(logits, label_ids)
                acc += tmp_accuracy
                
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                if self.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = self.learning_rate * warmup_linear(global_step/num_train_steps, self.warmup_proportion)
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
                self.score_qa(val_dataloader,val_index,data["val"],model,"val",ep,method)
            if test_dataloader_list:
                for index,tup in enumerate(test_dataloader_list):
                    self.score_qa(tup[0],tup[1],data["test"][index],model,"test"+str(index),ep,method)            
            
            print("After Current-Epoch:",self.best_metric)
            
        return model,self.best_metric
        
        
    def predict_qa(self,eval_dataloader,eval_index,data,model,typ,ep,method="sum"):      
        device = self.device
        model.eval()
        nb_eval_steps, nb_eval_examples = 0, 0
        all_results_sum = {}
        all_results_max = {}
        is_multifact =False
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
                scores = softmax(scores)
                if ":" in idx:
                    qid = idx.split(":")[0]
                    is_multifact = True
                else:
                    qid = idx
                if qid not in all_results_sum:
                    all_results_sum[qid]=scores
                    all_results_max[qid]=scores
                else:
                    current_scores = all_results_sum[qid]
                    all_results_sum[qid]=(np.array(scores) + np.array(current_scores)).tolist()
                    
                    current_scores = all_results_max[qid]
                    final_scores=[]
                    for cscore,pscores in zip(scores,current_scores):
                        maxscore = max(cscore,pscores)
                        final_scores.append(maxscore)
                    all_results_max[qid]=final_scores
                    
        accuracy_dict = {}
        count = 0
        
        corrects_sum={}
        incorrects_sum={}
        corrects_max={}
        incorrects_max={}
        
        for qid,scores in all_results_sum.items():
            maxidx = self.get_score(data,qid,scores)
            qidx = qid if not is_multifact else qid+":0"
            if self.get_label(data,qid) == maxidx:
                count+=1
                corrects_sum[qidx]=maxidx
            else:
                incorrects_max[qidx]={"pred":maxidx,"scores":scores,"label":self.get_label(data,qid),"row_pred":data[qid+":"+str(maxidx)],}
                incorrects_sum[qidx]={"pred":maxidx,"scores":scores,"label":data[qidx].label,"row_pred":data[qid+":"+str(maxidx)],"row_label":data[qid+":"+str(data[qidx].label)]}
        accuracy_dict["sum_acc"] = float(count)/float(len(all_results_sum))      
        
        count = 0
        for qid,scores in all_results_max.items():
            maxidx = self.get_score(data,qid,scores)
            qidx = qid if not is_multifact else qid+":0"
            if self.get_label(data,qid) == maxidx:
                count+=1
                corrects_max[qid]=maxidx
            else:
                incorrects_max[qidx]={"pred":maxidx,"scores":scores,"label":self.get_label(data,qid),"row_pred":data[qid+":"+str(maxidx)],}
                incorrects_sum[qidx]={"pred":maxidx,"scores":scores,"label":data[qidx].label,"row_pred":data[qid+":"+str(maxidx)],"row_label":data[qid+":"+str(data[qidx].label)]}
        accuracy_dict["max_acc"] = float(count)/float(len(all_results_max))      
        
        logger.info(json.dumps(accuracy_dict))
        
        print("Saving Score Files!!")
        ofd = open(self.output_dir+"/"+typ+"all_predicts.txt","w+")
        res = {"max":all_results_max,"sum":all_results_sum}
        ofd.write(json.dumps(res))
        ofd.close()
        
        return accuracy_dict,incorrects_sum,incorrects_max,all_results_max,all_results_sum
    
    def get_score(self,data,qid,scores):
        idxs = np.argsort(scores).tolist()
#         return idxs[-1]
        if qid+":"+str(idxs[-1]) in data:
            return idxs[-1]
        elif qid+":"+str(idxs[-2]) in data:
            return idxs[-2]
        elif qid+":"+str(idxs[-3]) in data:
            return idxs[-3]
        elif qid+":"+str(idxs[-4]) in data:
            return idxs[-4]
    
    def get_label(self,data,qid):
        for idx in range(0,4):
            if qid+":"+str(idx) in data:
                return data[qid+":"+str(idx)].label
    
    def score_qa(self,eval_dataloader,eval_index,data,model,typ,ep,method="sum"):      
        device = self.device
        model.eval()
        nb_eval_steps, nb_eval_examples = 0, 0
        tq = tqdm(eval_dataloader, desc="Scoring:")
        all_results_sum = {}
        all_results_max = {}
        is_multifact = False
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
                scores = softmax(scores)
                if ":" in idx:
                    qid = idx.split(":")[0]
                    is_multifact=True
                else:
                    qid = idx
                if qid not in all_results_sum:
                    all_results_sum[qid]=scores
                    all_results_max[qid]=scores
                else:
                    current_scores = all_results_sum[qid]
                    all_results_sum[qid]=(np.array(scores) + np.array(current_scores)).tolist()
                    
                    current_scores = all_results_max[qid]
                    final_scores=[]
                    for cscore,pscores in zip(scores,current_scores):
                        maxscore = max(cscore,pscores)
                        final_scores.append(maxscore)
                    all_results_max[qid]=final_scores
        
        accuracy_dict = {}
        count = 0
        
        corrects_sum={}
        incorrects_sum={}
        corrects_max={}
        incorrects_max={}
        
        for qid,scores in all_results_sum.items():
            maxidx = np.argsort(scores).tolist()[-1]
            qidx = qid if not is_multifact else qid+":0"
            if self.get_label(data,qid) == maxidx:
                count+=1
        accuracy_dict["sum_acc"] = float(count)/float(len(all_results_sum))      
        
        count = 0
        for qid,scores in all_results_max.items():
            maxidx = np.argsort(scores).tolist()[-1]
            qidx = qid if not is_multifact else qid+":0"
            if self.get_label(data,qid) == maxidx:
                count+=1
        accuracy_dict["max_acc"] = float(count)/float(len(all_results_max)) 
        
        acc = max(accuracy_dict["sum_acc"],accuracy_dict["max_acc"])
        
        if self.best_metric is None:
            self.best_metric = {}
            self.best_metric["best-acc"] = acc
        
        self.best_metric["%s-%d"%(typ,ep)]=accuracy_dict
        
        if self.best_metric["best-acc"] <= acc and typ == "val":
                # Save Best Val Accuracy Model
                self.best_metric["best-acc"] = acc
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(self.output_dir, "best_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                
        logger.info(json.dumps(accuracy_dict))
        return 

    
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]
        
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()
        
class InputExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 idx,
                 context_sentence,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.swag_id = idx
        self.context_sentence = context_sentence
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"swag_id: {self.swag_id}",
            f"context_sentence: {self.context_sentence}",
            f"ending_0: {self.endings[0]}",
            f"ending_1: {self.endings[1]}",
            f"ending_2: {self.endings[2]}",
            f"ending_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

tokenmap = {}
def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    indexmap = {}
    passagelens = []
    sum_of_labels = 0
    for example_index, example in tqdm(enumerate(examples),desc="Tokenizing"):
        if example.context_sentence in tokenmap:
            context_tokens = tokenmap[example.context_sentence]
        else:
            context_tokens = tokenizer.tokenize(example.context_sentence)
            tokenmap[example.context_sentence] = context_tokens
        #start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            
            if ending in tokenmap:
                ending_tokens = tokenmap[ending]
            else:
                ending_tokens = tokenizer.tokenize(ending)
                tokenmap[ending] = ending_tokens
                
            passagelens.append(len(context_tokens_choice)+len(ending_tokens)+3)
            
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        indexmap[example_index]=example.swag_id
        sum_of_labels+=label
        
#         if example_index < 5:
#             logger.info("*** Example ***")
#             logger.info(f"swag_id: {example.swag_id}")
#             for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
#                 logger.info(f"choice: {choice_idx}")
#                 logger.info(f"tokens: {' '.join(tokens)}")
#                 logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
#                 logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
#                 logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
#             if is_training:
#                 logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id = example_index,
                choices_features = choices_features,
                label = label
            )
        )

    
    print("Passage Token Lengths Distribution", passagelens[-1], np.percentile(passagelens,50), np.percentile(passagelens,90), np.percentile(passagelens,95), np.percentile(passagelens,99))
    assert sum_of_labels > 0
    return features,indexmap


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