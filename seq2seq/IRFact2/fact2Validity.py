import  json
#import org.allenai.nlpstack.postag.defaultPostagger
#from allenai.nlpstack.postag import defaultPostagger
#import nltk
#nltk.download('averaged_perceptron_tagger')
import csv
import os



def main(srcPredFile, srcIRFile,order):
    genFact2Dict = {}
    for data in open(srcPredFile,'r'):
        line = json.loads(data)
        #print (line)

        src = line['metadata']['source_tokens']
        hypo = src[0:src.index("@@SEP@@")]
        fact1 = src[src.index("@@SEP@@")+1:]
        target = line['metadata']['target_tokens']
        pred1 = line['predicted_tokens'][0]
        pred2 = line['predicted_tokens'][1]
        pred3 = line['predicted_tokens'][2]

        key = "".join(hypo)+"".join(fact1)
        #key = "".join(hypo)

        '''print("hypo:",hypo)
        print("fact1:",fact1)
        print("target:",target)
        print("pred1:",pred1)
        print("pred2:",pred2)
        print("pred3:",pred3)
        print("key:",key)'''

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

        pred = [i.lower() for i in pred1]
        scores.append(len(set(pred))/(1.0*len(pred)))
        pred = [i.lower() for i in pred2]
        scores.append(len(set(pred))/(1.0*len(pred)))
        pred = [i.lower() for i in pred3]
        scores.append(len(set(pred))/(1.0*len(pred)))

        # Case : 2
        #

        #postaggedTokens = defaultPostagger.postagTokenized(pred1)
        #print (postaggedTokens)

        '''tokens = nltk.word_tokenize(" ".join(pred1))
        print(nltk.pos_tag(tokens))

        tokens = nltk.word_tokenize(" ".join(pred2))                            
        print(nltk.pos_tag(tokens))

        tokens = nltk.word_tokenize(" ".join(pred3))                            
        print(nltk.pos_tag(tokens))
        '''
        line["scores"] = scores

        #print(line["scores"])
        topScore = max(scores)
        topPred = line['predicted_tokens'][scores.index(topScore)]

        #print(topScore, topPred)

        #13-441 and 14-643, 13-613 and 9-623 have same hypothesis and fact1 and fact2 so had to do this
        if key in genFact2Dict:
            #print ("Duplicate:",key)
            key+="1"
            #bananascanbemadefromthemovingwinds.windisasourceofenergy
            #Ariverbankismadeofloam.soilisarenewableresourceforgrowingplants
            if key in genFact2Dict:
                #print("Agan Duplicate",key)
                key+="1"
                if key in genFact2Dict:
                    #print("Third Dup",key)
        genFact2Dict[key] = (topScore, topPred)

    #mergeToOrigFile("irdataset/test-trained.tsv",genFact2Dict)
    #mergeToOrigFile("irdataset/train.tsv",genFact2Dict)
    #mergeToOrigFile("irdataset/val-trained.tsv",genFact2Dict)


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
        #print (l)
        if order == 0:
            hypo = l[1]
            fact1 = l[2]
        elif order == 1:
            hypo = l[2]
            fact1 = l[1]
        else:
            print("Unknown Order: ",order)

        join = hypo.replace(" ","")+fact1.replace(" ","")
        #join = hypo.replace(" ","")
        #print (join)
        #print(genFact2Dict[join])
        '''for j in genFact2Dict[str]:
            pred = " ".join(j)
            l.append(pred)
            #13-441 and 14-643, 13-613 and 9-623 have same hypothesis and fact1 and fact2 so had to do this
            if str+"1" in genFact2Dict:
                #print (l)
                #print (genFact2Dict[str])
                #print (genFact2Dict[str+"1"])
                genFact2Dict[str] = genFact2Dict[str+"1"]
        '''
        #writer.writerow(l)

        if join in genFact2Dict:
            c+=1
            score = genFact2Dict[join][0]
            fact2 = " ".join(genFact2Dict[join][1])
            l.append(fact2)
            l.append(score)
            if join+"1" in genFact2Dict:
                genFact2Dict[join] = genFact2Dict[join+"1"]
                if join+"11" in genFact2Dict:
                    genFact2Dict[join+"1"] = genFact2Dict[join+"11"]

        writer.writerow(l)

    print(c)
    #input("Wait:")


if __name__ == "__main__":

    ## 0 - Hypo Fact1 correct order
    ## 1 - Fact1 Hypo Reverse order
    '''srcPredFile = "copynet_test_IR.json"
    srcIRFile = "irdataset/test-trained.tsv"
    main(srcPredFile,srcIRFile,0)'''
    '''srcPredFile = "copynet_train_IR.json"
    srcIRFile = "irdataset/train.tsv"
    main(srcPredFile,srcIRFile,0)'''
    '''srcPredFile = "copynet_val_IR.json"
    srcIRFile = "irdataset/val-trained.tsv"
    main(srcPredFile,srcIRFile,0)'''
    srcPredFile = "copynet_top10merged_test_IR.json"
    srcIRFile = "irdataset/top10-merged-test.tsv"
    main(srcPredFile,srcIRFile,1)
    srcPredFile = "copynet_top10merged_val_IR.json"
    srcIRFile = "irdataset/top10-merged-val.tsv"
    main(srcPredFile,srcIRFile,1)

