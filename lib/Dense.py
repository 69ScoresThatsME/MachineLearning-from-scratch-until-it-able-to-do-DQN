import numpy as np
from lib.Perceptron import *

class Dense:

    def __init__(self,size,activate_function="relu")->None:
        self.size=size
        self.activate_function=activate_function
        self.perceptrons=np.array([Perceptron(activate_function=activate_function) for _ in range(size)])
        self.input=np.zeros(size)
        self.sum_array=np.zeros(size)
        self.training_round=0
        
    def insert(self,input_feature)->None:
        compare=np.zeros(2)

        if type(input_feature)!=type(compare):
            input_feature=np.array(input_feature)
        
        self.input=input_feature.copy()

        for perceptron in self.perceptrons:
            perceptron.insert(self.input)

    def run(self):

        idx=0
        self.sum_array=np.zeros(self.size)
        for perceptron in self.perceptrons:
            self.sum_array[idx]=perceptron.run()
            idx+=1

        # if self.activate_function=="softmax":
        #     e=np.exp(self.sum_array-np.max(self.sum_array))
        #     self.sum_array=e/np.sum(e)
        
        return self.sum_array
    
    def apply_gradient(self):
        for perceptron in self.perceptrons:
            perceptron.apply_gradient()

    def backward(self,delta):
        
        gradiant=np.zeros(len(self.input))
        
        for i in range(self.size):
            gradiant+=self.perceptrons[i].backward(delta[i])
        
        return gradiant
        
    def train(self,target,show=False,step=1000000):
        self.training_round+=1
        idx=0
        for perceptron in self.perceptrons:
            perceptron.train(target[idx],step)
            idx+=1

        if show:
                print(f"======================= {self.training_round=} ======================= {self.sum_array=}") 
           
        