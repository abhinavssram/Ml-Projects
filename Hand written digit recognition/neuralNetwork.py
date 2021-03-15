#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:21:31 2019

@author: abhinavsamala
"""
import numpy as np

# Extracting data from csv files and creating input and output vectors .............
nl=input("no of layers :")
nhu=input("no of units in hidden layer :")
epoch=14
csvf=np.genfromtxt('/Users/abhinavsamala/Documents/ACADS/pypgms/assignments/mnist_train.csv',delimiter=",")
   
y= np.asfarray(csvf[:, :1])
    
x=np.asfarray(csvf[:, 1:])
    #x=np.dtype(int)    
NormC=255*0.99 + 0.01
x=np.asfarray(x)/NormC
#x=np.array(x,ndmin=2).T
nl=int(nl)
nhu=int(nhu)
labelvec=np.arange(10)

exvec=(labelvec==y).astype(np.float)
exvec[exvec==0]=0.01
exvec[exvec==1]=0.99



csvtf=np.genfromtxt('/Users/abhinavsamala/Documents/ACADS/pypgms/assignments/mnist_test.csv',delimiter=",")


xt=np.asfarray(csvtf[:, 0:])
xt=np.asfarray(xt)/NormC

def relu(z):

    return 1.0/(1.0+ np.exp(-z))
def relu_der(z):

    x=1/(1+np.exp(-z))
    dx=(x)*(1-x)
    return dx

# Neural network class ........................

class neuralNetwork:

# Initilizing variables     
    def initial (self,x,y,l,lu):
        self.x=x
        self.y=y
        self.l=l

        self.l=int(l)
        lu=int(lu)
        
        self.iniweights(l,lu)

# Initilisizing weigths 
            
    def iniweights (self,l,lu):    
        
        self.w=[]
        k=0
        while(k<(l+1)):
            if (k==0):
                c1=784
                c2=lu
                wm=np.random.randn(c2,c1)
                self.w.append(wm)
            elif (k==l):
                c1=lu
                c2=10
                wm=np.random.randn(c2,c1)
                self.w.append(wm)
            else:
                c1=lu
                c2=lu
                wm=np.random.randn(c2,c1)
                self.w.append(wm)
                

            k=k+1  

        
#  Forward and bacward propagation       
        
    def feedandback (self,x,y):

        x = np.array(x, ndmin=2).T
        temp=[x]

        k=0
        while(k<(self.l+1)):
    
            inv=temp[-1]

            x=np.dot(self.w[k],inv)
            self.otv=relu(x)
            temp.append(self.otv)
            k=k+1
        

#        backpropagation starts.........................

        
        m=self.l+1
        y=np.array(y,ndmin=2).T
        Oerr= y-self.otv
        
        while (m>1):
                self.otv=temp[m]

                inv=temp[m-1]
                tmp = Oerr * self.otv*(1-self.otv)
                
                tmp = np.dot(tmp, inv.T)

                self.w[m-1]=self.w[m-1]+(0.098)*tmp
                Oerr = np.dot(self.w[m-1].T,Oerr)
                m=m-1

# To test the "Test data"
    def test(self,invec):

        invect = np.array(invec, ndmin=2).T
        kk = 1
        
        while  (kk<(self.l+2)) :

            
            uu = np.dot(self.w[kk-1],invect)

            outvec = relu(uu)
#            print('kk: ',kk,'outvec: ',outvec)
            invect = outvec
#            print('outvec: ',outvec.__sizeof__)
            
            kk=kk + 1

        return outvec
        
    

if __name__ == "__main__": 
    

    Nnobj=neuralNetwork()
    Nnobj.initial(x,y,nl,nhu)
    
    for k in range (epoch):
        for i in range(len(y)):
            Nnobj.feedandback(x[i],exvec[i])
    jj=0
    for jj in range(len(xt)):

            res=(Nnobj.test(xt[jj]))
            print(np.argmax(res))
            
       
def main():
    main()    