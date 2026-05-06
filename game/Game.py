import numpy as np
from typing import Any
import random
# import math
import os

class Game:

    def __init__(self):
        self.map=[]
        self.px=0
        self.py=0
        self.wx=0
        self.wy=0
        self.ppx=0
        self.ppy=0
        self.winning=False
        self.move=0
        self.over=False
        self.backtrack=[]


    def create_1(self):
        map=[]
        for i in range(10):
            temp=[]
            for j in range(10):
                temp.append(0)
            map.append(temp)
        self.map=map

    def create_2(self):
        self.wx=random.randint(1,9)
        self.wy=random.randint(1,9)

        self.map[self.wx][self.wy]=7
        self.map[0][0]=6

    def get_state(self):
        flat=np.array(self.map).flatten()

        # dx=self.wx-self.px
        # dy=self.wy-self.py
        return np.append(flat,[self.wx,self.wy,self.px,self.py])

    def get_reward(self):
        x=self.wx
        y=self.wy-1
        
        if self.move>=30:
            self.over=True
            return -10

        if self.map[self.px][(self.py+1)%10]==7:
            self.winning=True
            return 100-(self.move*0.1)
        else:
            prev_dist=(((self.ppx-x)**2)+((self.ppy-y)**2))**0.5
            curr_dis=(((self.px-x)**2)+((self.py-y)**2))**0.5
            progress=(prev_dist-curr_dis)*0.5

     
            return progress-0.05
           

    def action(self,input_val):
        self.ppx=self.px
        self.ppy=self.py
        if input_val==1:
            self.py=(self.py-1)%10
        elif input_val==2:
            self.px=(self.px-1)%10
        elif input_val==3:
            self.py=(self.py+1)%10
        elif input_val==4:
            self.px=(self.px+1)%10
        self.map[self.ppx][self.ppy]=0
        self.map[self.wx][self.wy]=7
        self.map[self.px][self.py]=6
        
        self.move+=1
        for i in range(len(self.backtrack)):
            if self.backtrack[i]==(self.px,self.py):
                self.over=True
                return self.get_state(),-10,self.winning
        self.backtrack.append((self.px,self.py))
        return self.get_state(),self.get_reward(),self.winning


    def play(self):
        self.create_1()
        self.create_2()

    def play_interface(self):
        self.create_1()
        self.create_2()

        while True:
            for i in self.map:
                for j in i:
                    print(f"{j} ",end="")
                print()    
            a,b,c=self.action(int(input("enter: ")))
            print(f"{b}     {c}")
            # os.system("cls")
            if self.winning: 
                for i in self.map:
                    for j in i:
                        print(f"{j} ",end="")
                    print()
                a,b,c=self.action(int(input("enter: ")))
                print(f"{b}     {c}")   
                os.system("cls")
                break
        print("Winningggggggggg")
        

