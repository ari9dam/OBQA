import os
import csv
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
import spacy

def dataPreparation(file,nlp,statsOut):
    """
        Prepares the Data given a file for Machine Translation Task  
    """
    tsvFileName = os.path.join("../processedData/Fact2_Only_F1_H_similar_tokens/", os.path.basename(file).split(".")[0]+"Seq2seq.tsv")
    print (tsvFileName)
    tsvFile = open(tsvFileName, 'w', newline='\n')
    writer = csv.writer(tsvFile, delimiter='\t')

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]

    removedFact2 = []

    #Medium space model taken for word similarity
    #nlp = spacy.load('en_core_web_md')
    similarityThreshold = 0.7
    linecnt = 0

    for each in data[1:]:
        linecnt +=1
        if linecnt%50 == 0:
            print("Line:",linecnt)
        id = each[header.index('id')]
        fact1 = each[header.index('fact1')]
        fact2 = each[header.index('Fact2')]
        ans = each[header.index('answerKey')]
        ansHypo = each[header.index(ans+"_Hypothesis")]

        inp = ansHypo + " @@SEP@@ " + fact1
        #out = fact2

        '''print (ansHypo)
        print (fact1)
        print (fact2)
        print ('1',WordTokenizer().tokenize(ansHypo))
        print ('2',WordTokenizer().tokenize(fact1))
        print ('3',WordTokenizer().tokenize(fact2))
        '''
        inputTokens = [i.text for i in WordTokenizer().tokenize(ansHypo)+WordTokenizer().tokenize(fact1)]
        inputTokens = [i for i in set(inputTokens)]
        outputTokens = WordTokenizer().tokenize(fact2)
        outputTokens = [i.text for i in outputTokens]
        #print("Input:",inputTokens)
        #print("Out:",outputTokens)
        strOutput = ""
        count = 0
        for word in outputTokens:
            if word in inputTokens:
                #print(word)
                strOutput += word+" "
                count+=1
            else:
                #print(nlp("Buy"))
                #print(nlp("buying").similarity(nlp("Buy")))
                spacyOutWord = nlp(word)
                similarityScore = [spacyOutWord.similarity(nlp(w)) for w in inputTokens]
                #print(similarityScore)
                if (max(similarityScore) >= similarityThreshold):

                    strOutput += inputTokens[similarityScore.index(max(similarityScore))]+" "
                    #print(inputTokens[similarityScore.index(max(similarityScore))])
                    #print(word,inputTokens[similarityScore.index(max(similarityScore))],max(similarityScore))

                    count+=1

        if count < 2:
            removedFact2.append(id)
            #print(strOutput)
            continue
        #if count == 0:
            #strOutput = fact2

        #input("WAIT")

        #fact2ProcessedTokens.append(count)

        #print ('4',strOutput.strip())
        #input("WAITYY")
        out = strOutput.strip()
        #print (id,"***",fact1,"***",fact2,"***",ans,"***",ansHypo,each[header.index('A_Hypothesis')],"***",each[header.index('B_Hypothesis')],"***",each[header.index('C_Hypothesis')],"***",each[header.index('D_Hypothesis')])

        writer.writerow([inp,out,id])
    tsvFile.close()

    #statsOut = open("statsOut.txt","w+")
    statsOut.write("Rejected Instances in "+tsvFileName+": "+str(len(removedFact2))+"\n")
    for i in removedFact2:
        statsOut.write(i+",")
    statsOut.write("\n")
    #---------- print data stats ---------
    #print ("Average tokens per data in file :",tsvFileName,"=",sum(fact2ProcessedTokens)/len(fact2ProcessedTokens))

    return tsvFileName



def main():

    trainFile = "../srcData/trainData.csv"
    validFile = "../srcData/devData.csv"
    testFile = "../srcData/testData.csv"
    #Medium space model taken for word similarity                               
    nlp = spacy.load('en_core_web_md')
    statsOut = open("../processedData/Fact2_Only_F1_H_similar_tokens/statsOut.txt","w+")

    trainSeq2SeqFile = dataPreparation(trainFile,nlp,statsOut)
    validSeq2SeqFile = dataPreparation(validFile,nlp,statsOut)
    testSeq2SeqFile = dataPreparation(testFile,nlp,statsOut)

if __name__ == "__main__":
    main()
