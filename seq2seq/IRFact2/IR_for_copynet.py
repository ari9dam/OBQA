"""
Used to create ID, source_token, target_token data required for obqa_copynet_reader from IR full data
top10 files have fact1 before hypo and rest have reverse order, hence another order indicator
"""

import os
import csv


def dataPreparation(file, order=0):
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
        if order == 0:
            hypo = each[1]
            fact1 = each[2]
        elif order == 1:
            hypo =each[2]
            fact1 = each[1]
        else:
            print("Error in Input Order")


        inp = hypo + " @@SEP@@ " + fact1


        writer.writerow([inp,"-",id])

    tsvFile.close()

    return tsvFileName



def main():

    testFile = "irdataset/test-trained.tsv"
    validFile = "irdataset/val-trained.tsv"
    trainFile = "irdataset/train.tsv"
    toptenTestFile = "irdataset/top10-merged-test.tsv"
    toptenValidFile = "irdataset/top10-merged-val.tsv"
    trainSeq2SeqFile = dataPreparation(trainFile,0)
    validSeq2SeqFile = dataPreparation(validFile,0)
    testSeq2SeqFile = dataPreparation(testFile,0)
    toptenTestSeq2SeqFile = dataPreparation(toptenTestFile,1)
    toptenValSeq2SeqFile = dataPreparation(toptenValidFile,1)

if __name__ == "__main__":
    main()
