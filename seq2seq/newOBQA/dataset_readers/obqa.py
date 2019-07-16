from typing import Dict
import json
import csv
import spacy
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("obqa")
class ObqaDatasetReader(DatasetReader):
    """
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):

        #Medium space model taken for word similarity
        nlp = spacy.load('en_core_web_md')
        similarityThreshold = 0.7

        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            reader = csv.reader(file_path)
            data = list(reader)
        header = data[0]
        for each in data[1:]:            
            id = each[header.index('id')]
            fact1 = each[header.index('fact1')]
            fact2 = each[header.index('Fact2')]
            ans = each[header.index('answerKey')]
            ansHypo = each[header.index(ans+"_Hypothesis")]

            
            inp = ansHypo + " @@SEP@@ " + fact1

            inputTokens = [i.text for i in WordTokenizer().tokenize(ansHypo)+WordTokenizer().tokenize(fact1)]
            inputTokens = [i for i in set(inputTokens)]
            outputTokens = WordTokenizer().tokenize(fact2)
            outputTokens = [i.text for i in outputTokens]
        
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
                print(strOutput)
                #continue
            if count == 0:
                strOutput = fact2
            
            out = strOutput.strip()        
        

            yield self.text_to_instance(inp, out)


    @overrides
    def text_to_instance(self, hypo_fact1: str, fact2: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_hypoFact1 = self._tokenizer.tokenize(hypo_fact1)
        tokenized_fact2 = self._tokenizer.tokenize(fact2)
        hypoFact1_field = TextField(tokenized_hypoFact1, self._token_indexers)
        fact2_field = TextField(tokenized_fact2, self._token_indexers)
        
        fields = {"hypo_fact1":hypoFact1_field, "fact2":fact2_field}
        return Instance(fields)








