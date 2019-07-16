import os
import csv
import torch
import torch.optim as optim
import itertools
import sys
sys.path.append('../')

import dataPreperation.Fact2_Only_F1_H_exact_tokens as data
#from dataPreperation.Fact2_Only_F1_H_exact_tokens import dataPreparation
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
#from obqa_datasetreader import Seq2SeqDatasetReader
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder

from obqa_seq2seq import SimpleSeq2Seq
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention


from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SimpleSeq2SeqPredictor



#Size of output
ENC_EMBEDDING_DIM = 256
TGT_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
CUDA_DEVICE = 0
numEpochs = 30
beamSize = 8

def findExtraVocab(data):

    allExtraVocab = []
    for i in range(len(data)):
        srcData = list(data[i]['source_tokens'])
        srcData = [str(i) for i in srcData]
        tgtData = list(data[i]['target_tokens'])
        tgtData = [str(i) for i in tgtData]
        #print(srcData,tgtData)
        extra = set(tgtData) - set(srcData)
        for j in extra:
            allExtraVocab.append(j)
        #print(allExtraVocab)
    #print (len(allExtraVocab))
    #print (len(set(allExtraVocab)))

    return allExtraVocab

def main():

    trainFile = "../srcData/trainData.csv"
    validFile = "../srcData/devData.csv"
    testFile = "../srcData/testData.csv"
    trainSeq2SeqFile = data.dataPreparation(trainFile)
    validSeq2SeqFile = data.dataPreparation(validFile)
    testSeq2SeqFile = data.dataPreparation(testFile)
    print (testSeq2SeqFile)
    #TokenIndexer Determines how string tokens gets represented as arrays of indexes in a model
    #SingleIdTokenIndexer = Tokens are single integers
    #TokenCharactersIndexer = Tokens as a list of integers
    # Read a tsvfile with paired instances (source, target)
    reader = Seq2SeqDatasetReader(
        source_tokenizer = WordTokenizer(),
        target_tokenizer = WordTokenizer(), # Defaults to source_tokenizer
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer()} # Defaults to source_token_indexers
    )

    # Each of the dataset is a list of each tokens (source_tokens, target_tokens)
    train_dataset = reader.read(trainSeq2SeqFile)
    validation_dataset = reader.read(validSeq2SeqFile)
    test_dataset = reader.read(testSeq2SeqFile)

    # Finding extra fact2 vocab
    trainExtraVocab = findExtraVocab(train_dataset)
    validExtraVocab = findExtraVocab(validation_dataset)
    testExtraVocab = findExtraVocab(test_dataset)
    finalExtraVocab = list(set(trainExtraVocab+validExtraVocab+testExtraVocab))
    print("length:",len(finalExtraVocab))
    #input()

    #vocab = Vocabulary.from_instances(train_dataset + validation_dataset, min_count={'tokens': 3, 'target_tokens': 3})
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset + test_dataset)
    # Train + Valid = 9703
    # Train + Valid + Test = 10099


    print ("Vocab SIze :",vocab.get_vocab_size('tokens'))

    encEmbedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=ENC_EMBEDDING_DIM)

    # Embedding for tokens since in the dataset creation time it is mentioned tokens
    source_embedder = BasicTextFieldEmbedder({"tokens": encEmbedding})

    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(ENC_EMBEDDING_DIM,HIDDEN_DIM,batch_first=True,dropout=0.2))


    attention = DotProductAttention()

    max_decoding_steps = 4  # TODO: make this variable
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim = TGT_EMBEDDING_DIM,
                          #target_namespace = 'target_tokens',
                          attention = attention,
                          beam_size = beamSize,
                          use_bleu = True,
                          extra_vocab = finalExtraVocab)
    #Can also specify lr=0.001
    optimizer = optim.Adam(model.parameters())

    # Data Iterator that specify how to batch our dataset
    # Takes data shuffles it and creates fixed sized batches
    #iterator = BasicIterator(batch_size=2)
    #iterator.index_with(vocab)
    # Pads batches wrt max input lengths per batch, sorts dataset wrt the fieldnames and padding keys provided for efficient computations
    iterator = BucketIterator(batch_size=50, sorting_keys=[("source_tokens", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model = model,
                      optimizer = optimizer,
                      iterator = iterator,
                      train_dataset = train_dataset,
                      validation_dataset = validation_dataset,
                      #patience = 3,
                      num_epochs = numEpochs,
                      cuda_device = CUDA_DEVICE)

    trainer.train()
    predictor = SimpleSeq2SeqPredictor(model, reader)

    '''for i in range(2):
        print ("Epoch: {}".format(i))
        trainer.train()

        predictor = SimpleSeq2SeqPredictor(model, reader)


        for instance in itertools.islice(validation_dataset, 10):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])
            """'{'predictions': [[1, 4, 5, 92, 8, 6, 1, 8, 6, 26, 3]], 
             'loss': 5.9835076332092285,
             'class_log_probabilities': [-20.10894012451172],
             'predicted_tokens': ['@@UNKNOWN@@', 'is', 'a', 'type', 'of', 'the', '@@UNKNOWN@@', 'of', 'the', 'sun']}
             """
            print (predictor.predict_instance(instance))
    '''

    outFile = open("output_"+str(HIDDEN_DIM)+"_"+str(numEpochs)+"_"+str(beamSize)+".csv","w")
    writer = csv.writer(outFile,delimiter="\t")
    for instance in itertools.islice(test_dataset,500):
        src = instance.fields['source_tokens'].tokens
        gold = instance.fields['target_tokens'].tokens
        pred = predictor.predict_instance(instance)['predicted_tokens']
        writer.writerow([src,gold,pred])


    outFile.close()
    

if __name__ == "__main__":
    main()
