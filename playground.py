from lib.Perceptron import *
from lib.Dense import *
from lib.Model import *
import numpy as np


a=[[1,1],[1,0],[0,1],[0,0]]
qa=[[0],[1],[1],[0]]

select=0

# d=Dense(size=2,activate_function="relu",learning_rate=0.5)
# d.insert(np.array(a[select]))
# d.train(np.array([qa[select],3]),show=True,step=100)

# print(d.run())

m=Model()
m.addDense(size=4,activate_function="tanh")
m.addDense(size=1,activate_function="sigmoid")

m.fit(np.array(a),np.array(qa),show=True,round=-1,clear=True)

print(m.predict(np.array(a)))