# OBQA
open book question answering



# Data Set
OBQA

## Parsing and Tagging of Wh Questions
1. Parser and Tagger are taken from https://github.com/tdozat/Parser-v3.
2. Parser and Tagger are trained using CoNLL 2018 dataset :

    * Git clone the repo. Create the Data Directory : data/CoNLL18/UD_English-EWT. Save both the Datasets and the embeddings.
    * Training Data : http://universaldependencies.org/conll18/
    * Word2Vec Embeddings : https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1989 . Only English Word2Vec embeddings are needed.
    * Environment : TensorFlow=1.4, Scipy, Matplotlib, Psutil, Python=3.6, Pandas, Conllu
    * Training Command : python main.py train ParserNetwork / TaggerNetwork
    * Run Model Command : python main.py --save_dir=\\$PATH_TO_NETWORK run \\$INPUTFILE --output_dir=\\$OUTPUTDIR
    * Key Point to Note: CoNLLU format needs to be adhered strictly, Tabs between columns.
    * Trained Models to be pushed at a location : [DropboxLocation]
    
## Word Intersection, Union and Seq2Seq Abductive IR 
1. Present in notebooks and folders

## Re-Ranking using SpaCY
1. Present in filterir

## BERT QA models
1. Runner and scorer are present
    
