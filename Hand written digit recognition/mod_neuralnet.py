#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:20:15 2019

@author: abhinavsamala
"""
import numpy as np

# Extracting data from csv files and creating input and output vectors .............
#nl=input("no of layers :")
#nhu=input("no of units in hidden layer :")

csvf=np.genfromtxt('/Users/abhinavsamala/Downloads/Neural_data/Toy/train.csv',delimiter=",")
   
y= np.asfarray(csvf[:, -1])
    
x=np.asfarray(csvf[:, 0:2])
#x=np.dtype(int)    
#NormC=255*0.99 + 0.01
#x=np.asfarray(x)/NormC
x=np.array(x,ndmin=2)
temp=[x]
print("temp:",(temp[-1]))
print("x[0].shape:",x[0].shape)
file1 = open("param.txt","r+")
L = file1.readlines(0)
#print("x.shape:",x.shape)
print("L:",L)
myIntegers = [int(x) for x in L[4].split()]
#print("myIntegers:",len(myIntegers))
input_nodes = (x[0].shape[0])

myIntegers.insert(0,input_nodes)

myIntegers.append(1)
nl= len(myIntegers)
print("nl:",nl)
epoch = L[2]
learn_rate = L[1]
#labelvec=np.arange(2)
#exvec=(labelvec==y).astype(np.int)
#exvec[exvec==0]=0.01
#exvec[exvec==1]=0.99


#Activation function
def sigmoidf(z):

    return 1.0/(1.0+ np.exp(-z))
def sig_der(z):

    x=1/(1+np.exp(-z))
    dx=(x)*(1-x)
    return dx


# Neural network class ........................

class neuralNetwork:

# Initilizing variables     
    def initial (self,x,y,myIntegers,bias=None,learn_rate):
        self.x=x
        self.y=y
        self.l=myIntegers
        self.bias = bias
        self.learn_rate = learn_rate
        self.iniweights()
        self.inibias()

    def iniweights (self):    
        
        no_layers= len(myIntegers)
        self.w=[]
        
        k=1
        while(k<(no_layers)):
                c1=int(myIntegrs[k-1])
                c2=int(myIntegers[k])
                wm=np.zero(c2,c1)
                self.w.append(wm)
                k=k+1
        
    def inibias(self):
        if self.bias:
            bias_node =1
        else:
            bias_node =0
        
#Feedforward and backpropagation...........        

        def feedandback (self,x,y):

            x = np.array(x, ndmin=2).T
            temp=[x]
#            print("temp: ",temp)
            k=0
            while(k<(self.l+1)):
    
                inv=temp[-1]

                x=np.dot(self.w[k],inv)
                self.otv=sigmoidf(x)
                temp.append(self.otv)
                k=k+1
            
           
##        backpropagation starts.........................
#
#        
#        m=self.l+1
#        y=np.array(y,ndmin=2).T
#        Oerr= y-self.otv
#        
#        while (m>1):
#                self.otv=temp[m]
#
#                inv=temp[m-1]
#                tmp = Oerr * self.otv*(1-self.otv)
#                
#                tmp = np.dot(tmp, inv.T)
#
#                self.w[m-1]=self.w[m-1]+(0.098)*tmp
#                Oerr = np.dot(self.w[m-1].T,Oerr)
#                m=m-1
#
#if __name__ == "__main__": 
#    
#    
#
#    Nnobj=neuralNetwork()
#    Nnobj.initial(x,y,nl)
#    
#    for k in range (epoch):
#        for i in range(len(y)):
#            Nnobj.feedandback(x[i],y[i]) 
##    jj=0
##    for jj in range(len(xt)):
##
##            res=(Nnobj.test(xt[jj]))
##            print(np.argmax(res))
##            
##       
#def main():
#    main()    