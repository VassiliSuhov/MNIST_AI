# -*- coding: utf-8 -*-
import numpy as np
import os 
import json
BASE_DIR = os.path.dirname(os.path.realpath(__file__))[:-5]

def TestTrainSet():
    return 1
def sigmoid(x):
    return 1/(1 + 2.71 ** - x.sum())

def linear(x):
    return x

def relu(x):
    return max(0 , x)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
    # remove largest value
   # expx = np.exp(x - np.max(x))
    #return expx / expx.sum(axis=0, keepdims=True)
    return x / x.sum(axis=0, keepdims=True)



    
activationFunctions = [sigmoid , linear , tanh , softmax]


def compute_loss(y , realY):
    
    lenght = realY.shape[0]
    return  np.sum((y - realY) ** 2)


def createzeros(shape):
    return np.zeros(shape)
   
def createRandomParams(size ):
    return np.random.random(size)

def GetInputOfType(type_):
    input_ = input("?")
    try:
        input_ = type_(input_)
    except:
        print("give input of type " + str(type_))
        input_ = GetInputOfType(type_)
    return input_


class MultiProcessingModelController:
    def __init__(self , data_x , data_y , ModelClass):
        
        self.INIT_FILE = "model/init.json"
        return
    
    
    
    

class baseUtulitys:
    def __init__(self , paramIndex):
        self.DATA_FOLDER = "model/"
        self.BASE_FILENAME = "init.json"
        self.PARAMS_FOLDER = "params/"
        self.PARAMS_FILENAME = "params"
        self.PARAMS_ERROR_KEY = "Cost"
        self.current_params = ''
        if(not os.path.exists(BASE_DIR + self.DATA_FOLDER + self.BASE_FILENAME)):
           self.InitialParams()
           self.CreateBaseFolder()
        else:
            self.current_params = self.PARAMS_FILENAME + str(paramIndex) + ".json"
            self.LoadModel()
           
    
    def InitialParams(self):
        return 
    def CustomParams(self):
        return []
    
    
    def createRandomState(self):
        return '[]'
    
    def CreateBaseFolder(self):
        
        init_params = {}
        if not os.path.exists(BASE_DIR + self.DATA_FOLDER):
            
            os.makedirs(BASE_DIR + self.DATA_FOLDER)
            
            
        if not os.path.exists(BASE_DIR + self.DATA_FOLDER + self.PARAMS_FOLDER ):
            
            os.makedirs(BASE_DIR + self.DATA_FOLDER + self.PARAMS_FOLDER)
        
        print("how many samples of parameters do you need?")
        numberofSamples = int(input("integers input ? :  "))
        
        print("123")
        for num in range(0 , numberofSamples):
            file = open(BASE_DIR + self.DATA_FOLDER + self.PARAMS_FOLDER + self.PARAMS_FILENAME+ str(num) + ".json" , 'w')
            file.write(self.createRandomState())
            file.close()
        
        print("aa")
       
        init_params[self.PARAMS_ERROR_KEY] = [10] * numberofSamples
        customParams = self.CustomParams()
        
        for param in customParams:
            init_params[param[0]] = param[1]
            
        if(not os.path.exists(BASE_DIR + self.DATA_FOLDER + self.BASE_FILENAME)):
            file = open(BASE_DIR + self.DATA_FOLDER + self.BASE_FILENAME, 'w')
            
            file.write(json.dumps(init_params))
            file.close()
        
        
        
            
        return 
    
    def LoadSample(self , index):
        file = open(BASE_DIR + self.DATA_FOLDER + self.PARAMS_FOLDER + self.PARAMS_FILENAME+ str(index) + ".json" , 'r')
        
        string  = file
        
        file.close()
        
        return string
    
    def TurnModelToString(self):
        return
    
    
    def TurnStringToModel(self , params , model_settings):
        return
    
    def SaveModel(self):
        file = open(BASE_DIR + self.DATA_FOLDER + self.PARAMS_FOLDER + self.current_params , 'w')
        
        file.write(self.TurnModelToString())
        file.close()
        
        return
      
    
    def LoadModel(self):
        init = open(BASE_DIR +self.DATA_FOLDER + self.BASE_FILENAME , 'r')
        file = open(BASE_DIR + self.DATA_FOLDER + self.PARAMS_FOLDER + self.current_params  , 'r')
        self.TurnStringToModel(file , init)
        file.close()
        init.close()
        return
    
        
    
