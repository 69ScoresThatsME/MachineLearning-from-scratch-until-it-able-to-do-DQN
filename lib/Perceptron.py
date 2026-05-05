import numpy as np
import math


class Perceptron:

    def __init__(self,activate_function="relu") -> None:
        self.input=np.array([])
        self.weight=np.array([])
        self.bias=0
        self.activate_function=activate_function
        self.sum=0
        self.training_round=0
        self.sum_weight_np=np.array([])
        self.sum_pre_activation=0
        self.m=np.zeros(self.weight.size)
        self.v=np.zeros(self.weight.size)
        self.m_bias=0
        self.v_bias=0
        self.t=0
        self.learning_rate=0.001
        self.grads=0
        self.grads_b=0
        self.batch_count=0

    def status(self,weight=False,bias=False):
        print(f"{self.input.size}")
        print(f"{self.activate_function=}")
        print(f"{self.training_round=}")
        print(f"output value={self.sum}")
        if weight:
            print(f"{self.weight=}")
        if bias:
            print(f"{self.bias}")

    def insert(self,input_feature)->None:
        compare=np.zeros(2)

        if type(input_feature)!=type(compare):
            input_feature=np.array(input_feature)
        
        self.input=input_feature.copy()
        
        if self.weight.size!=self.input.size:
            self.weight=np.random.uniform(-1,1,self.input.size)
            self.sum_weight_np=np.ones(self.input.size)
            self.m=np.zeros(self.weight.size)
            self.v=np.zeros(self.weight.size)
            self.grads=np.zeros(self.input.size)

    def sum_weight(self):
        self.sum=0
        self.sum_weight_np=self.weight*self.input
        self.sum=np.sum(self.sum_weight_np)+self.bias

    def activation(self):
        self.sum_pre_activation=self.sum
        if self.activate_function=="relu":
            if self.sum<0:
                self.sum=0

        elif self.activate_function=="sigmoid":
            self.sum=1/(1+(math.e**(-self.sum)))

        elif self.activate_function=="step":
            if self.sum>=0:
                self.sum=1
            else:
                self.sum=0

        elif self.activate_function=="tanh":
            self.sum=np.tanh(self.sum)

        elif self.activate_function=="leaky":
            if self.sum<=0:
                self.sum*=0.01
        #else:

        #     print(f"Invalid Input **{self.activate_function}** I'm going to relu!!")
        #     if self.sum<0:
        #         self.sum=0


        return self.sum

    def activation_derivation(self):
        z=self.sum_pre_activation

        if self.activate_function=="relu":
            return 1 if z>0 else 0

        elif self.activate_function=="sigmoid":
            # self.activation()
            return self.sum*(1-self.sum)

        elif self.activate_function=="step":
            return 0

        elif self.activate_function=="tanh":
            return 1-(np.tanh(z)**2)

        elif self.activate_function=="leaky":
            return 0.01 if z<0 else 1
        else:
            return 0

    def run(self):
        self.sum_weight()
        self.activation()

        return self.sum
    
    def apply_gradient(self):

        #Avg Gradiant for Grad
        self.grads/=self.batch_count
        self.grads_b/=self.batch_count

        # Adam
        self.t+=1

        self.m=(0.9*self.m)+(0.1*self.grads)
        self.v=(0.999*self.v)+(0.001*(self.grads**2))
        self.m_bias=(0.9*self.m_bias)+(0.1*(self.grads_b))
        self.v_bias=(0.999*self.v_bias)+(0.001*(self.grads_b)**2)

        #Adam: bias correction?
        m_hat=self.m/(1-0.9**self.t)
        v_hat=self.v/(1-0.999**self.t)
        m_hat_b=self.m_bias/(1-0.9**self.t)
        v_hat_b=self.v_bias/(1-0.999**self.t)

        self.weight+=(self.learning_rate*m_hat)/(np.sqrt(v_hat)+1e-8)
        self.bias+=(self.learning_rate*m_hat_b)/(np.sqrt(v_hat_b)+1e-8)


        # reset gradiant
        self.grads=np.zeros_like(self.weight)
        self.grads_b=0
        self.batch_count=0
        
    def backward(self,delta):
        d=delta*self.activation_derivation()
        gradient = d * self.weight 

        self.grads+=d*self.input
        self.grads_b+=d
        self.batch_count+=1

        return gradient

    def train(self,target,show=False,step=1000000):
        self.run()
        while target!=self.sum and self.training_round<=step:
            self.training_round+=1
            error=target-self.sum

            self.weight+=(error*self.input*self.learning_rate)
            self.bias+=(error*self.learning_rate)
            self.run()      #<-------------------------------------------------------------
            if show:
                print(f"===== {self.weight=} ====== {self.training_round=} =========== {self.sum=}")
                



