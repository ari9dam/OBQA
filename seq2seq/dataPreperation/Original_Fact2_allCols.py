import os
import csv
#from allennlp.data.tokenizers.word_tokenizer import WordTokenizer


def dataPreparation(file):
    """
        Prepares the Data given a file for Machine Translation Task  
    """
    tsvFileName = os.path.join("../processedData/Original_Fact2/", os.path.basename(file).split(".")[0]+"AllCols.tsv")
    print (tsvFileName)
    tsvFile = open(tsvFileName, 'w', newline='\n')
    writer = csv.writer(tsvFile, delimiter='\t')

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]

    fact2ProcessedTokens = []

    writer.writerow(['id','Question','A','B','C','D','fact1','Fact2','humanScore','answerKey','A_Hypothesis','B_Hypothesis','C_Hypothesis','D_Hypothesis','correctHypo','correctHypoWithFact1'])
    
    for each in data[1:]:
        id = each[header.index('id')]
        fact1 = each[header.index('fact1')]
        fact2 = each[header.index('Fact2')]
        ans = each[header.index('answerKey')]
        ansHypo = each[header.index(ans+"_Hypothesis")]

        inp = ansHypo+fact1+fact2
        joinCond=inp.replace(" ","")
        each.append(ansHypo)
        each.append(joinCond)
        #print (each)
        #print(joinCond)

        #input("WAUT")
        writer.writerow(each)
    tsvFile.close()

    #---------- print data stats ---------
    #print ("Average tokens per data in file :",tsvFileName,"=",sum(fact2ProcessedTokens)/len(fact2ProcessedTokens))

    return tsvFileName



def main():

    trainFile = "../srcData/trainData.csv"
    validFile = "../srcData/devData.csv"
    testFile = "../srcData/testData.csv"
    trainSeq2SeqFile = dataPreparation(trainFile)
    validSeq2SeqFile = dataPreparation(validFile)
    testSeq2SeqFile = dataPreparation(testFile)

if __name__ == "__main__":
    main()
