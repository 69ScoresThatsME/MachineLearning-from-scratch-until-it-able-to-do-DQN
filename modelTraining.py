from lib.Perceptron import *
from lib.Dense import *
from lib.Model import *
import numpy as np
from game.Game import *
import random
import copy
model=Model()
model.addDense(128,activate_function="sigmoid")
model.addDense(64,activate_function="sigmoid")
model.addDense(4,activate_function="linear")

target=Model()
target.addDense(128,activate_function="sigmoid")
target.addDense(64,activate_function="sigmoid")
target.addDense(4,activate_function="linear")

epoch=100002
gamma=0.95

game=Game()
game.play()
buffer=[]
current_epoch=0



load=100000
model.load(f"models/model_{load}.pkl")
target.load("models/model_90000.pkl")
loaded=True
current_epoch=load

epsilon=1.0
epsilon_min=0.1
epsilon_decay=0.2
if not loaded:
    epsilon_decay=0.9999


while True:
    

    state=game.get_state()
    action:int

    if np.random.random()<epsilon:
        action=random.randint(1,4)
    else:
        q=model.predict([state])
        action=int(q[0].argmax()+1)

    next_state,reward,winning=game.action(action)
    # next_q=target.predict(next_state)
    

    if game.winning or game.over:
        buffer.append((state,action,reward,next_state,True))
        game=Game()
        game.play()
    else:
        buffer.append((state,action,reward,next_state,winning))

    if len(buffer)>10000:
        buffer.pop(0)

        batch_size=32

        batch=random.sample(buffer,batch_size)

        states=np.array([b[0] for b in batch]).reshape(batch_size,104)
        actions=np.array([b[1] for b in batch])
        rewards=np.array([b[2] for b in batch])
        next_states=np.array([b[3] for b in batch]).reshape(batch_size,104)

        qs=model.predict(states)
        # print(qs.shape)
        next_qs=target.predict(next_states)

        epsilon=max(epsilon*epsilon_decay,epsilon_min)

        for i in range(batch_size):
            if batch[i][4]:
                qs[i][actions[i]-1]=rewards[i]
            else:
                qs[i][actions[i]-1]=rewards[i]+(gamma*np.max(next_qs[i]))
        current_epoch+=1
        os.system("cls")
        model.fit(states,qs,epoch=1,show=True)
        print(f"{current_epoch}/{epoch} ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(f"{epsilon=}")
        if loaded:
            if current_epoch%1000==0 and current_epoch!=load:
                model.save(f"models/model_{current_epoch+load}.pkl")
        # if current_epoch in [10,50,100,500,1000,2000,3000,4000,5000,6200,7500,8400,10000,20000,30000
        #                       ,40000,50000,60000,70000,80000,90000,100000]:
        
        #     model.save(f"models/model_{current_epoch}.pkl")
        if current_epoch>=load+epoch:
            break
    if current_epoch%100==0 and current_epoch!=0 and current_epoch!=load:
        target.denses=copy.deepcopy(model.denses)