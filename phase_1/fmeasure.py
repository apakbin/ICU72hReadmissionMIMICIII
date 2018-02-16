import numpy as np
import pandas as pd

def fscore_precision_recall_specificity(tp, fp, tn, fn):
    fscore=0
    precision=0
    recall=0
    specificity=0
    
    if tp+fn != 0:
        recall=float(tp)/(tp+fn)
    
    if tp+fp != 0:
        precision=float(tp)/(tp+fp)
    
    if tn+fp != 0:
        specificity=float(tn)/(tn+fp)
    
    if precision+recall !=0:
        fscore=float(2*(precision*recall))/(precision+recall)
    
    return [fscore,precision,recall,specificity]

def tp_fp_tn_fn(predicted, labels):
    tp=0
    fp=0
    tn=0
    fn=0
    
    for i in range(0,len(predicted)):
        predicted_=predicted[i]
        label_=labels[i]
        if predicted_==label_:
            if predicted_==1:
                tp=tp+1
            else:
                tn=tn+1
        else:
            if predicted_==1:
                fp=fp+1
            else:
                fn=fn+1

    return [tp,fp,tn,fn]

def roc(predicted, labels, number_of_points=100):
    min_value=min(predicted)
    max_value=max(predicted)
    
    values=np.linspace(min_value,max_value,number_of_points)
    
    ROC=pd.DataFrame(index=range(0,number_of_points),columns=['threshold', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'specificity', 'fscore'])
    for i in range(0,len(values)):
        [TP,FP,TN,FN]=tp_fp_tn_fn(predicted>values[i],labels)
        [fscore,precision,recall,specificity]=fscore_precision_recall_specificity(TP, FP, TN, FN)
        
        ROC.iloc[i,:]=[values[i],TP,FP,TN,FN,precision,recall,specificity,fscore]
    return ROC

def maximize_roc(roc, maximization_criteria):
    to_be_maximized=roc[maximization_criteria].values
    maximum_index=np.argmax(to_be_maximized)
    return roc.iloc[[maximum_index]]