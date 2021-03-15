#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:58:34 2019

@author: abhinavsamala
"""

import numpy as np

dataset=np.genfromtxt('/Users/abhinavsamala/Documents/ACADS/pypgms/assignments/assign-2/datasets/haberman-data.csv',delimiter=",")
testdata=np.genfromtxt('/Users/abhinavsamala/Documents/ACADS/pypgms/assignments/assign-2/datasets/habermantest.csv',delimiter=",")


x=np.array(dataset[:,0:3])
y=np.array(dataset[:,3])

xt=np.array(testdata[:,0:3])
yt=np.array(testdata[:,3])
#print('xt',xt)



def normalise(x):
    mins = np.min(x) 
    maxs = np.max(x) 
#    print('maxs',maxs)
#    print('mins',mins)
    rng = maxs - mins 
    norm_x =  ((x-np.mean(x))/rng)
    return norm_x
x=normalise(x)

def sigmoid(z):

    return 1.0/(1.0+ np.exp(-z))

import math
class logisticRegression():
    
    def outputcal(self,beta,x,n):
#       exout = np.ones((x.shape[0],1))
#       paracoeff=paracoeff.reshape(1,n+1)
#       print('paracoeff-35',paracoeff)
#       for i in range (0,x.shape[0]):
#           exout[i]=1/(1+ np.exp(-(float(np.matmul(paracoeff, x[i])))))
#           
#       exout = exout.reshape(x.shape[0])
#       print('exout-40',exout)
#       return exout
        
        exout = np.ones((x.shape[0],1))
        beta = beta.reshape(1,n+1)
#        print('beta-45',beta)
#        print('x-46',x)
        for i in range(0,x.shape[0]):
#            print('np.matmul(beta, x[i])',np.exp(-float(np.matmul(beta, x[i]))))
            exout[i] = 1 / (1 + math.exp(-float(np.matmul(beta, x[i]))))
        exout = exout.reshape(x.shape[0])
#        print('exout-48',exout)
        return exout
    
        
    def train(self,x,y,a):
        n=x.shape[1]
        o=np.ones((x.shape[0],1))
        x=np.concatenate((o,x),axis=1)
        #parametres initialisation
        self.beta=np.zeros(n+1)    
        #Finding expected output
        exout=self.outputcal(self.beta,x,n)

        #gradientdescentalgo & updation
#        b = b + alpha * (y – prediction) * prediction * (1 – prediction) * x
        self.final_beta,self.exout=self.updation(self.beta,a,exout,x,y,n)
#        print('exout',self.outputcal(self.final_beta,x,n))
#        print('y',y)
        return self.final_beta,self.exout
    
    def updation(self,beta, a, exout, x, y, n):
        
        
#        print('beta-60',beta)
        beta[0] = beta[0] - (a/x.shape[0]) * sum(exout - y)
        for j in range(1,n+1):
                beta[j]=beta[j]-(a/x.shape[0])*sum((exout-y)*x.transpose()[j])
#        print('beta-64',beta)        
        exout = self.outputcal(beta, x, n)
#        print('exout-67',exout)
        beta = beta.reshape(1,n+1)
        return beta,exout 
    
#    def test(self,xt,parameters,yt):
#        n=xt.shape[1]
#        o=np.ones((xt.shape[0],1))
#        xt=np.concatenate((o,xt),axis=1)
#        predictions = self.outputcal(parameters,xt,n)
#        print('predictions',predictions)
#        print('yt',yt)
#        accuracy
#        correct=0
#        for i in range(0,yt.shape[0]):
#            if (predictions[i]>=0.5):
#                predictions[i]=1
#            elif (predictions[i]<0.5):
#                predictions[i]=2
#            
#        for k in range(0,yt.shape[0]):
#            if(predictions[i]==yt[i]):
#                correct=correct+1
#            
#        accuracy = correct/(yt.shape[0]-1) 
#        print('accuracy :',accuracy)

if __name__ == "__main__": 
    

    obj=logisticRegression()
    
    
    
    parameters,exout=obj.train(x,y,0.01)
#    pp= obj.outputcal(parameters,x,x.shape[1])
#    print('pp',pp)
#    print('y',y)
#    print('y-exout',y-exout)
#    print('para-102',parameters)
    xt=normalise(xt)
#    print('xt',xt)
    n=xt.shape[1]
    o=np.ones((xt.shape[0],1))#
    xt=np.concatenate((o,xt),axis=1)
    

#    print('n',xt.shape[1])
    predictions = obj.outputcal(parameters,xt,n)
#    print('predictions-141',predictions)
#    print('y-exout',yt-predictions)
#        accuracy
    correct=0
    for i in range(0,yt.shape[0]):
            if (predictions[i]>0.5025):
                predictions[i]=1
            else:
                predictions[i]=2
                
#    print('predictions-151',predictions[0]==yt[0])        
    for k in range(0,yt.shape[0]):
            if(predictions[k]==yt[k]):
                correct=correct+1
            
    accuracy = correct/(yt.shape[0]) 
    print('accuracy :',accuracy)   
    

#   precision calculations
    tp =0
    fp = 0

    for j in range(0, yt.shape[0]):
        if predictions[j] == yt[j] == 1:
            tp = tp + 1
        elif predictions[j] == 1 and yt[j] == 2:
            fp = fp + 1
    precision = tp/(tp + fp)
    print('precision :',precision)         
#   recall calculation
    
    fn=0            
    for jj in range(0,yt.shape[0]):
        if predictions[jj] == 2 and yt[jj]==1 :
            fn=fn+1
    recall=tp/(tp+fn)        
    print('recall :',recall)    
#  F1-score
    
    f1score= (2*precision*recall)/(precision+recall)      
    print('f1score :',f1score)
    
    
    
def main():
    main()         