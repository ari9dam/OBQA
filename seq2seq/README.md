# MyOBQA
Open Book Question Answering My Codes

### Method
* Source data 
* Train Model with newOBQA/data
  * CUDA_VISIBLE_DEVICES=0 python run.py train experiments/final14.json --include-package models --include-package dataset_readers --include-package metrics -s output_final14
  * Check the overlap_metric
* Prediction :
  * IR Files having all rows source (5 files) -> IRFact2/irdataset/
  * IRFact2/IR_for_copynet.py - converts  IRFact2/irdataset/* files and prepares as input to copynet_obqa (source_token, target_token, src_id) produces IRFact2/irdataset/\*Seq2Seq files
  * python run.py predict output_final14/model.tar.gz ../IRFact2/irdataset/val-trainedSeq2seq.tsv --predictor seq2seq --use-dataset-reader --output-file IR_new_val.json --include-package models --include-package dataset_readers
  * All prediction outputs in IR_ouputs/IR_new_\*.json files
* Merge Back to IR Original data

### Data :
Seq2Seq/
#### Fact2_Only_F1_H_exact_tokens
* For each entry removed all exact tokens of Hypothesis and Fact1 from Fact2
* Removed all Facts where noTokens < 2
* F2 Intersect (F1 Union H)
* No restrictions in Vocabulary
* Average tokens per data in Train file :  4.2629482071713145
* Average tokens per data in Valid file :  4.278301886792453
* Average tokens per data in Test  file :  4.102564102564102

Seq2Seq_Restricted/
* For each entry removed all exact tokens of Hypothesis and Fact1 from Fact2
* Removed all Facts where noTokens < 2
* F2 Intersect (F1 Union H)
* Vocab Restriction : Using H and F1_i translate to all tokens in H and F1_i 
* Average tokens per data in Train file :  4.2629482071713145
* Average tokens per data in Valid file :  4.278301886792453
* Average tokens per data in Test  file :  4.102564102564102
                    
* Result :
beamSize=8
01/28/2019 20:11:30 - INFO - allennlp.training.trainer -   gpu_0_memory_MB |  1791.000  |       N/A
01/28/2019 20:11:30 - INFO - allennlp.training.trainer -   gpu_1_memory_MB |    11.000  |       N/A
01/28/2019 20:11:30 - INFO - allennlp.training.trainer -   loss            |     1.982  |     8.096
01/28/2019 20:11:30 - INFO - allennlp.training.trainer -   cpu_memory_MB   |  2973.600  |       N/A
01/28/2019 20:11:30 - INFO - allennlp.training.trainer -   BLEU            |       N/A  |     0.000
01/28/2019 20:11:30 - INFO - allennlp.training.trainer -   Epoch duration: 00:00:04
* Outputs containing other words may be because of Beam Search

beamSize=0
01/28/2019 20:23:49 - INFO - allennlp.training.trainer -   gpu_1_memory_MB |    11.000  |       N/A
01/28/2019 20:23:49 - INFO - allennlp.training.trainer -   cpu_memory_MB   |  2975.292  |       N/A
01/28/2019 20:23:49 - INFO - allennlp.training.trainer -   loss            |     1.974  |     8.287
01/28/2019 20:23:49 - INFO - allennlp.training.trainer -   BLEU            |       N/A  |     0.000
01/28/2019 20:23:49 - INFO - allennlp.training.trainer -   gpu_0_memory_MB |  1807.000  |       N/A
01/28/2019 20:23:49 - INFO - allennlp.training.trainer -   Epoch duration: 00:00:04
* Outputs with @@PADDING@@


