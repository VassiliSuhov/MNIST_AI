# -*- coding: utf-8 -*-
import numpy as np
import os 
import json
BASE_DIR = os.path.dirname(os.path.realpath(__file__))[:-5]

def TestTrainSet():
    return 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def BackPropSigmoid(ForwardGradients, outputs , inputs):
    return ForwardGradients *  (1 - outputs)




def linear(x):
    return x

def relu(x):
    return max(0 , x)

def BackPropRelu(outputsGrads , outputs , inputs):
    outputsGrads[outputsGrads < 0 ] = 0
    return outputsGrads

def BackpropTanh(outputsGrads , outputs , inputs):
    return (1 - np.power(outputs , 2)) * outputsGrads
    
def tanh(x):
    return np.tanh(x)

def softmax(x):
    # remove largest value
    exps = np.exp(x)
    return exps / np.sum(exps)




def BackPropSoftmax(outputGradients , outputs ,inputs):
     n = np.size(outputs)
     return np.dot((np.identity(n) - outputs.T) * outputs, outputGradients)





def CalculateLayerDerivative(forwardGradients, weights , inputs):
    #gradients of the weights
    grad_weights = forwardGradients[0] * inputs
    for i in range(1 ,len(forwardGradients)):
        grad_weights = np.vstack((grad_weights , inputs * forwardGradients[i]))
    
    inputs_grad = weights[0] * forwardGradients[0]
    for i in range(1 ,len(forwardGradients)):
        inputs_grad = np.vstack((inputs_grad , weights[i] * forwardGradients[i]))
        
    inputs_grad = np.sum(inputs_grad , axis=0)
    #deleting the last input grad because it containes the 1
    return[ np.delete(inputs_grad, len(inputs_grad) - 1) , grad_weights]

activationNames = ["sigmoid" , "linear" , "tanh" , "softmax" , "relu"]

activationFunctions = [sigmoid , linear , tanh , softmax , relu]

backPropFunctions = [BackPropSigmoid , 1 , BackpropTanh , BackPropSoftmax , BackPropRelu]


def compute_loss(y , realY):
    
    
    return  np.sum((y - realY) ** 2)

def BackpropagateLossFunction(outputs , Realy):
    #so the derivative of the loss is 1
    #then the way you get the loss is (y - realY) * (y - realY)
    #so the derivative is (y - realY)
    return (outputs - Realy)


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
    
    
"""  
a = np.array([3, 6 , 2])
b = np.array([[5 , 5 , 5 , 5] , [6 , 6, 6, 6] , [ 7, 7 , 7 , 7]])
c = np.array([9,9,9,1])



d = CalculateLayerDerivative(a, b, c)
print(d)
"""
