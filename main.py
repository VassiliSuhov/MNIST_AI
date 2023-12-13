# -*- coding: utf-8 -*-

import keras
from aiLib.aiLib import *
import numpy as np
DATASET_PATH = "C:/Users/sukho/OneDrive/Documents/DataSets/mnist/mnist.npz"



def LoadDataSetMNIST():
    return keras.datasets.mnist.load_data(path = DATASET_PATH)

def LoadTrainSet():
    Limit = 40000
    dataset =  keras.datasets.mnist.load_data(path = DATASET_PATH)
    return dataset[0][0][:Limit] , dataset[0][1][:Limit]


def Format(x):
    new_x = []
    for image in x:
        new_list = np.array([])
        for row in image:
            new_list = np.concatenate((new_list , row  / 255) )
        new_x.append(new_list)
    return new_x

def ForamtYMNIST(y):
    
    entrie = Utilitys.createzeros(10)
    entrie[y[0]] = 1 
    new_y = np.array([entrie])
    
    for num in range(1 , len(y)):
        entrie = Utilitys.createzeros(10)
        entrie[y[num]] = 1
        new_y = np.vstack((new_y , entrie))
    
    return new_y

def loadtestSet():
    dataset =  keras.datasets.mnist.load_data(path = DATASET_PATH)
    return dataset[1][0] , dataset[1][1]
    


x , y = LoadTrainSet()

x = Format(x)

y_ = ForamtYMNIST(y)


x_test = x[1]
y_test = y[1]

model = NeuralNetwork(5)


data_layers = model.OptimizeAmount(100, x, y_, 0.1 , 1000)



def Testparams(x , y):
    wins = 0
    loss = 0
    for index in range(0 , len(x)):
        prediction = model.ComputeModel(x[index], model.layers)
        
        highest_num = 0
        highest_index = 0
        
        for output in range(0 , len(prediction)):
            if(prediction[output] > highest_num):
                highest_num = prediction[output]
                highest_index = output
        if(highest_index == y[index]):
            wins +=1
        else:
            loss += 1
    print("wins : ")
    print(wins)
    print("losses :")
    print(loss)
    print("accuracy")
    print(wins / (wins + loss))
    

