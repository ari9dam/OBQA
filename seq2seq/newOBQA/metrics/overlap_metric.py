from typing import Tuple                                                        
                                                                                
from overrides import overrides                                                 
                                                                                
from allennlp.training.metrics.metric import Metric                             
                                                                                
                                                                                
@Metric.register("overlap_metric")                                        
class OverlapMetric(Metric):                                             
                                                                                
    def __init__(self) -> None:                                                 
        self._total_match = 0.0                                                 
        self._count = 0                                                         
                 
    @overrides                                                                  
    def __call__(self, pred, gold):                                             
                        
        nBatchMatched = 0                        
        nBatchGoldWords = 0                   
        
        for (eachInstancePred, eachInstanceGold) in zip(pred,gold):
            eachInstanceGold = [each.lower() for each in eachInstanceGold]
            eachInstanceGold = set(eachInstanceGold)
            # Finding the best prediction out of 3 which have maximum fractional overlap with the gold
            scores = []                                                             
            for each in eachInstancePred:                                               
                predfact2 = [word.lower() for word in each]
                predfact2 = set(predfact2)                              
                scores.append(len(predfact2.intersection(eachInstanceGold))/(1.0* len(eachInstanceGold)))
            eachInstancePred = eachInstancePred[scores.index(max(scores))]                       


            nMatched = 0
            nWordsGold = len(eachInstanceGold)
            for eachWordPred in set(eachInstancePred) :
                if(eachWordPred in eachInstanceGold):
                    nMatched += 1
            nBatchMatched += nMatched
            nBatchGoldWords += nWordsGold
        self._total_match += nBatchMatched
        self._count += nBatchGoldWords                                                                       
                                                                        
    @overrides                                                                  
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:           
                                                                                
        overlap = (self._total_match / self._count) if self._count > 0 else 0      
        if reset:                                                               
            self.reset()                                                        
        return {"overlap_metric":overlap}                                                          
                                                                                                      
    @overrides                                                                  
    def reset(self):                                                            
        self._total_match = 0.0                                                 
        self._count = 0                                                         
                                                                                
                                                                                
    def __str__(self):                                                          
        return f"OverlapMetric(overlap_metric={self._total_match}, count={self._count})"                  
                                                                                
                          
