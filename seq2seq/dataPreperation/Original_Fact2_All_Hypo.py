import os
import csv


def dataPreparation(file):
    """
        Prepares the Data given a file for Machine Translation Task  
    """
    tsvFileName = os.path.join("../processedData/Original_Fact2_All_Hypo/", os.path.basename(file).split(".")[0]+"Seq2seq.tsv")
    print (tsvFileName)
    tsvFile = open(tsvFileName, 'w', newline='\n')
    writer = csv.writer(tsvFile, delimiter='\t')

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]

    
    for each in data[1:]:
        id = each[header.index('id')]
        fact1 = each[header.index('fact1')]
        fact2 = each[header.index('Fact2')]
        ans = each[header.index('answerKey')]
        ansHypo = each[header.index(ans+"_Hypothesis")]


        inp = ansHypo + " @@SEP@@ " + fact1
        out = fact2


        allhypo = ['A_Hypothesis','B_Hypothesis','C_Hypothesis','D_Hypothesis']
        allhypo.remove(ans+"_Hypothesis")
        #print(allhypo)

        # Writing for correct options
        writer.writerow([inp,out])
        # Writing for other options
        for otherOptions in allhypo:
            inp = each[header.index(otherOptions)]+ " @@SEP@@ " + fact1
            writer.writerow([inp, out])


    tsvFile.close()

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
