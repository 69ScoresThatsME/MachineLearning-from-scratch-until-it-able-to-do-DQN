import numpy as np
import math

class Conv2d:

    def __init__(self,kernel_size:int,kernels:int):
        self.kernel_size=kernel_size
        self.kernels=[]
        self.amount=kernels

    def __create_kernel(self,in_chs):
        self.kernels=[]
        for i in range(self.amount):
            kernel_gen=np.random.randn(in_chs,self.kernel_size,self.kernel_size)*np.sqrt(2/(self.kernel_size**2*in_chs))#<----scale
            self.kernels.append(kernel_gen)

    def __kerneling(self,input_arr,kernel):
        return input_arr*kernel
    
    def run(self,input_arr):

        if input_arr.ndim==2:
            input_arr=input_arr[np.newaxis,:,:]

        in_chs,input_h,input_w=input_arr.shape

        if len(self.kernels)==0:
            self.__create_kernel(in_chs)
        output_list=[]
        

        

        for kernel in self.kernels:
            out_h=input_h-self.kernel_size+1
            out_w=input_w-self.kernel_size+1
            output=np.zeros((out_h,out_w))

            for r in range(out_h):
                for c in range(out_w):

                    for ch in range(in_chs):
                        window=input_arr[ch,r:r+self.kernel_size,c:c+self.kernel_size]
                        output[r,c]+=np.sum(self.__kerneling(window,kernel[ch]))

            output_list.append(output)

        return np.array(output_list)