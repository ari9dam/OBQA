import json
import os
import csv

def mergeData(fact2File,srcFile):
    data = []
    genFact2Dict = {}
    c=0
    for line in open(fact2File,'r'):
        c+=1
        line = json.loads(line)
        src = line['metadata']['source_tokens']
        hypo = src[0:src.index("@@SEP@@")] 
        fact1 = src[src.index("@@SEP@@")+1:]

        inp = "".join(hypo)+"".join(fact1)
 
        target =  line['metadata']['target_tokens']
        goldFact2 = "".join(target)
        pred1 = line['predicted_tokens'][0]
        pred2 = line['predicted_tokens'][1] 
        pred3 = line['predicted_tokens'][2]
        key = inp+goldFact2
        #13-441 and 14-643, 13-613 and 9-623 have same hypothesis and fact1 and fact2 so had to do this
        if key in genFact2Dict:
            key+="1"
        genFact2Dict[key] = (pred1,pred2,pred3)
    #print (c)
    print (len(genFact2Dict))

    tsvFileName = os.path.join("", os.path.basename(srcFile).split(".")[0]+"gen.tsv")
    print (tsvFileName)
    tsvFile = open(tsvFileName, 'w', newline="\n")
    writer = csv.writer(tsvFile, delimiter="\t")

    with open(srcFile, 'r') as f:
        reader = csv.reader(f,delimiter="\t")
        data = list(reader)
    header = data[0]
    header.append('predictedFact1')
    header.append('predictedFact2')
    header.append('predictedFact3')
    writer.writerow(header)
    c=0
    for l in data[1:]:
        c+=1
        str = l[-1]
        for j in genFact2Dict[str]:
            pred = " ".join(j)
            l.append(pred)
            #13-441 and 14-643, 13-613 and 9-623 have same hypothesis and fact1 and fact2 so had to do this
            if str+"1" in genFact2Dict:
                #print (l)
                #print (genFact2Dict[str])
                #print (genFact2Dict[str+"1"])
                genFact2Dict[str] = genFact2Dict[str+"1"]

        writer.writerow(l)
    print(c)

def main():
    fileName = "copynet_test_novocab.json"
    srcFile = "../../processedData/Original_Fact2/testDataAllCols.tsv"
    mergeData(fileName,srcFile)
    fileName = "copynet_train_novocab.json"
    srcFile = "../../processedData/Original_Fact2/trainDataAllCols.tsv"
    mergeData(fileName,srcFile)
    fileName = "copynet_dev_novocab.json"
    srcFile = "../../processedData/Original_Fact2/devDataAllCols.tsv"
    mergeData(fileName,srcFile)


if __name__ == "__main__":
    main()



