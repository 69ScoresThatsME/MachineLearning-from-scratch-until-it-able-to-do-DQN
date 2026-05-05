from lib.Perceptron import *
from lib.Dense import *
from typing import Any
import numpy as np
import os

class Model:

    def __init__(self):
        self.denses=[]
        self.input=np.array([])
        self.dense_count=0

    def addDense(self,size,activate_function="relu"):
            self.denses.append(Dense(size,activate_function=activate_function))
            self.dense_count+=1
            return self

    def insert(self,input_feature)->None:
        compare=np.zeros(2)

        if type(input_feature)!=type(compare):
            input_feature=np.array(input_feature)
        
        self.input=input_feature.copy()

    def run(self,show=False):
        current_input=self.input.copy()
        for i in range(len(self.denses)):
            dense=self.denses[i]
            dense.insert(current_input)
            new_input=dense.run()
            if show:
                 print(new_input)
            current_input=new_input

        return current_input
    
    def train(self,input,target,round=-1,show=False):
        self.insert(input)
        current_round=0
        MSE=0
        while current_round!=round:
            current_round+=1
            output=self.run()
            delta=target-output
            MSE=np.mean(delta**2)
            
            if show:
                if round==-1:
                    print(f"{current_round=} {MSE=}",end=" [")
                else :
                    print (f"{current_round=}/{round} {MSE=}",end=" [")

            if MSE<0.001 and round==-1:
                break

            for dense in reversed(self.denses):
                delta=dense.backward(delta)
                if show:
                    print("=",end="")
            if show:
                print("]")
        return MSE
    
    def fit(self,inputs,target,epoch=-1,show=False,clear=False,batch_size=1):

        current_epoch=0
        while current_epoch!=epoch:

            

            if clear:
                os.system("cls")

            if show:
                    if epoch==-1:
                        print(f"{current_epoch=} ",end=" [")
                    else :
                        print (f"{current_epoch=}/{epoch} ",end=" [")
            total_Mse=0
            current_epoch+=1
            for i in range(0,len(inputs),batch_size):

                inputs_batch=inputs[i:i+batch_size]
                targets_batch=target[i:i+batch_size]

                for i in range(len(inputs_batch)):
                    total_Mse+=self.train(inputs_batch[i],targets_batch[i],round=1)
                for dense in reversed(self.denses):
                    dense.apply_gradient()
                if show:
                    print(f"=",end="")
            if show:
                print("]")
                print(f"MSE={total_Mse/len(inputs)}")
            if total_Mse/len(inputs)<0.001 and epoch==-1:
                break
        
    def predict(self,inputs):
        output=[]
        for input in inputs:
            self.insert(input)
            output.append([self.run()])
        
        return np.array(output)


