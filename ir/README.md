## Fact IR from OpenBook

pip requirements - 

pytorch-pretrained-bert == 0.6.1

spacy == 2.0.16

torch >= 1.0


The code to train a STS Model on the modified data is present in run_sts and scoring is in run_sts_score. Download the data from the line
and put it in a data/ folder. 

To evaluate the rankings, there is a score_ir.py 

Already ranked data is present in the ranked folder, and source knowledge facts are present in knowledge.
Hypothesis folder contains the generated hypothesis data.


Link to Data:
https://drive.google.com/drive/folders/1XM9krxl7weUITTIAIVRZXPSOKodn4Ixu?usp=sharing
