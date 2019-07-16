import  json
import csv
import os



def main(srcPredFile, srcIRFile,order):
    genFact2Dict = {}
    for data in open(srcPredFile,'r'):
        line = json.loads(data)

        src_id = line['metadata']['src_id']

        # Case : 1
        # Calculated for cases where same noun get repeated both as subject and objects
        # Also handles same words repeated due to copynet
        # Lower Casing + ignore punctuation
        # electricity is electricity
        # Iron nail is iron
        # 'A', 'bat', 'is', 'a', 'bat'
        #['Water', 'can', 'turn', 'to', 'vapor', 'vapor', 'when', 'a', 'pot', 'energy']
        '''
            pred1: ['Water', 'can', 'turn', 'to', 'vapor', 'vapor', 'in', 'a', 'room', 'temperature', 'setting']
            pred2: ['Water', 'can', 'turn', 'to', 'vapor', 'vapor', 'in', 'a', 'room', 'temperature', 'setting', '.']
            pred3: ['Water', 'can', 'turn', 'to', 'vapor', 'vapor', 'in', 'a', 'room', 'temperature', 'temperature', 'setting']
            pred1: ['Water', 'can', 'turn', 'to', 'vapor', 'vapor']
            pred2: ['Water', 'can', 'turn', 'to', 'vapor', 'when', 'placing', 'water', 'in', 'a', 'freezer']
            pred3: ['Water', 'can', 'turn', 'to', 'vapor', 'vapor', 'when', 'placing', 'in', 'a', 'freezer']
        '''

        scores = []

        for eachPred in line['predicted_tokens']:
            pred = [i.lower() for i in eachPred]
            scores.append(round((len(set(pred))/(1.0*len(pred))),2))

        line["scores"] = scores

        topScore = max(scores)
        topPred = line['predicted_tokens'][scores.index(topScore)]

        genFact2Dict[src_id] = (topScore, topPred)


    #### MERGING INTO IR DATA ####


    tsvFileName = os.path.join("", os.path.basename(srcIRFile).split(".")[0]+"withFact2.tsv")
    print (tsvFileName)
    tsvFile = open(tsvFileName, 'w', newline="\n")
    writer = csv.writer(tsvFile, delimiter="\t")

    with open(srcIRFile, 'r') as f:
        reader = csv.reader(f,delimiter="\t")
        data = list(reader)

    c=0
    for l in data:
        sid = l[0]

        if sid in genFact2Dict:
            c+=1
            score = genFact2Dict[sid][0]
            fact2 = " ".join(genFact2Dict[sid][1])
            l.append(fact2)
            l.append(score)

        writer.writerow(l)

    print("Total records Merged : ",c)


if __name__ == "__main__":

    ## 0 - Hypo Fact1 correct order
    ## 1 - Fact1 Hypo Reverse order

    pathOfOutput = ""
    srcPredFile = pathOfOutput+"IR_new_test.json"
    srcIRFile = "../IRFact2/irdataset/test-trained.tsv"
    main(srcPredFile,srcIRFile,0)
    '''srcPredFile = pathOfOutput+"IR_new_copynet_train.json"
    srcIRFile = "../IRFact2/irdataset/train.tsv"
    main(srcPredFile,srcIRFile,0)'''
    srcPredFile = pathOfOutput+"IR_new_val.json"
    srcIRFile = "../IRFact2/irdataset/val-trained.tsv"
    main(srcPredFile,srcIRFile,0)
    srcPredFile = pathOfOutput+"IR_new_top10_test.json"
    srcIRFile = "../IRFact2/irdataset/top10-merged-test.tsv"
    main(srcPredFile,srcIRFile,1)
    srcPredFile = pathOfOutput+"IR_new_top10_val.json"
    srcIRFile = "../IRFact2/irdataset/top10-merged-val.tsv"
    main(srcPredFile,srcIRFile,1)
    
