import os
import csv
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer


def dataPreparation(file):
    """
        Prepares the Data given a file for Machine Translation Task  
    """
    tsvFileName = os.path.join("../processedData/Original_Fact2/", os.path.basename(file).split(".")[0]+"Seq2seq.tsv")
    print (tsvFileName)
    tsvFile = open(tsvFileName, 'w', newline='\n')
    writer = csv.writer(tsvFile, delimiter='\t')

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]

    fact2ProcessedTokens = []
    
    for each in data[1:]:
        id = each[header.index('id')]
        fact1 = each[header.index('fact1')]
        fact2 = each[header.index('Fact2')]
        ans = each[header.index('answerKey')]
        ansHypo = each[header.index(ans+"_Hypothesis")]

        inp = ansHypo + " @@SEP@@ " + fact1
        out = fact2

        '''print (ansHypo)
        print (fact1)
        print (fact2)
        print ('1',WordTokenizer().tokenize(ansHypo))
        print ('2',WordTokenizer().tokenize(fact1))
        print ('3',WordTokenizer().tokenize(fact2))
        '''
        '''inputTokens = [i.text for i in WordTokenizer().tokenize(ansHypo)+WordTokenizer().tokenize(fact1)]
        outputTokens = WordTokenizer().tokenize(fact2)
        strOutput = ""
        count = 0
        for word in outputTokens:
            if word.text in inputTokens:
                strOutput += word.text+" "
                count+=1

        if count < 2:
            continue

                                                
        fact2ProcessedTokens.append(count)

        #print ('4',strOutput.strip())
        #input("WAITYY")
        out = strOutput.strip()'''
        #print (id,"***",fact1,"***",fact2,"***",ans,"***",ansHypo,each[header.index('A_Hypothesis')],"***",each[header.index('B_Hypothesis')],"***",each[header.index('C_Hypothesis')],"***",each[header.index('D_Hypothesis')])

        writer.writerow([inp,out])
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
