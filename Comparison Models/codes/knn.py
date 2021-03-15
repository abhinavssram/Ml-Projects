#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:48:17 2019

@author: abhinavsamala
"""

import numpy as np
import math
import operator
dataset=np.genfromtxt('/Users/abhinavsamala/Documents/ACADS/pypgms/assignments/assign-2/datasets/haberman-data.csv',delimiter=",")
testdata=np.genfromtxt('/Users/abhinavsamala/Documents/ACADS/pypgms/assignments/assign-2/datasets/habermantest.csv',delimiter=",")



x=np.array(dataset[:,0:3])
y=np.array(dataset[:,3])

xt=np.array(testdata[:,0:3])
yt=np.array(testdata[:,3])

#def of l2 distance

def l2distance(x1,x2,f):
    dist=0
    for i in range(f):
        dist+=np.square((x1[i]-x2[i]))
    return np.sqrt(dist)

def knn(dataset,testdata,k):
    
    dists={}
    
    
    l=testdata.shape[1]
#    calculating distances b/w testdata and trainingdata
    
    for i in range(len(dataset)):
        dist=l2distance(testdata,dataset[i],l)
        dists[i]=dist[0]
#    print(len(dists))    
#   sort the distances in ascending order
    sorteddists=sorted(dists.items(),key=operator.itemgetter(1))
#   getting k neighbours
    neighbours=[]
    
    for j in range(k):
        neighbours.append(sorteddists[j][0])
#    finding the class it belongs to
        
    labellist={}    
    for m in range(len(neighbours)):
        res = dataset[neighbours[m]][-1]
 
        if res in labellist:
            labellist[res] += 1
        else:
            labellist[res] = 1

#   return predictec label
    sortedlabels = sorted(labellist.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedlabels[0], neighbours)          
# 
#   accuracy
def accuracy(neighbours,yt):
        correct = 0
        for x in range(len(neighbours)):
            if neighbours[x]== yt[x]: 
                correct= correct + 1
			
        print ('accuracy: ',correct/float(len(neighbours)))
#precision
#def precision(predictions,yt):        
#    tp =0
#    fp = 0
#
#    for j in range(0, len(predictions)):
#        if predictions[j] == yt[j] == 1:
#            tp = tp + 1
#        elif predictions[j] == 1 and yt[j] == 2:
#            fp = fp + 1
#    precision = tp/(tp + fp)
#    print('precision :',precision)
#    return tp,fp,precision         
##   recall calculation
#def recall(predictions,yt,tp,fp):    
#    fn=0            
#    for jj in range(0,len(predictions)):
#        if predictions[jj] == 2 and yt[jj]==1 :
#            fn=fn+1
#    recall=tp/(tp+fn)        
#    print('recall :',recall)
#    return recall    
##  F1-score
#def F1score(precision,recall):     
#    f1score= (2*precision*recall)/(precision+recall)      
#    print('f1score :',f1score)        
if __name__ == "__main__": 
    k=input("no of neighbours: ")
    k=int(k)
    
     
    result,neigh=knn(dataset,testdata,k) 
#    print(result)
#    print('neigh:',neigh)
    accuracy(result,yt)
#    tp,fp,precs=precision(result,yt)
#    resofrecal=recall(result,yt,tp,fp)
#    F1score(precs,resofrecal)
    
    
    