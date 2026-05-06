from lib.Model import *
from game.Game import *
import numpy as np
import time
import os

#Gemini Generate

model=Model()
model.addDense(128,activate_function="sigmoid")
model.addDense(64,activate_function="sigmoid")
model.addDense(4,activate_function="linear")
model.load("models/model_120000.pkl")

game=Game()
game.play()

while not game.winning:
    os.system("cls")

    for row in game.map:
        print(" ".join(map(str,row)))

    state=game.get_state()
    predictions=model.predict([state])[0]
    action=np.argmax(predictions)+1
    print(game.move)
    print(f"\nAI Predictions: {predictions}")
    print(f"AI Decided Action: {action}")
    

    game.action(action)
    

    time.sleep(0.3)


os.system("cls")
for row in game.map:
    print(" ".join(map(str,row)))
print("\n" + "="*20)
print(game.move)
print("Winningggggggggg")
print("="*20)