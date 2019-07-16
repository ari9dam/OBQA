import os
import csv


def dataPreparation(file):
    """
        Prepares the Data given a file for Machine Translation Task  
    """
    tsvFileName = os.path.join("irdataset/", os.path.basename(file).split(".")[0]+"Seq2seq.tsv")
    print (tsvFileName)
    tsvFile = open(tsvFileName, 'w', newline='\n')
    writer = csv.writer(tsvFile, delimiter='\t')

    with open(file, 'r') as f:
        reader = csv.reader(f,delimiter="\t")
        data = list(reader)

    
    for each in data:
        #print(each)
        id = each[0]
        #Change for merged IR DATA
        hypo = each[2]
        fact1 = each[1]


        inp = hypo + " @@SEP@@ " + fact1


        writer.writerow([inp,"-"])

    tsvFile.close()

    return tsvFileName



def main():

    #testFile = "irdataset/test-trained.tsv"
    #validFile = "irdataset/val-trained.tsv"
    #trainFile = "irdataset/train.tsv"
    toptenTestFile = "irdataset/top10-merged-test.tsv"
    toptenValidFile = "irdataset/top10-merged-val.tsv"
    #trainSeq2SeqFile = dataPreparation(toptenTestFile)
    validSeq2SeqFile = dataPreparation(toptenValidFile)
    testSeq2SeqFile = dataPreparation(toptenTestFile)

if __name__ == "__main__":
    main()
