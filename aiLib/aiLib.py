

import numpy as np
import math
from . import Utilitys


import json
#your path here 




class layer:
    def __init__(self , numberOfNodes , paramSize , activationFunction):
        
       self.params = Utilitys.createRandomParams([numberOfNodes , paramSize])
       self.activationType = activationFunction
       
    def computeLayer(self , inputs):
        
        result = np.matmul( self.params , inputs)
        
        return Utilitys.activationFunctions[self.activationType](result)
  
    
  
    
class simplemethod(Utilitys.baseUtulitys):
    def simplefinc(self):
        return
    
  
class NeuralNetwork(Utilitys.baseUtulitys):
    def __init__(self , paramFileIndex):
        
        self.layerSize = []
        self.layers = []
        self.PARAMS_ACTIVATION = "activtions_params"
        self.PARAMS_SHAPES = "shapes"
        Utilitys.baseUtulitys.__init__(self, paramFileIndex)
    
    
    
    
    
    #Whenever the init file does not exist , it calls this function before creating the init file
    def InitialParams(self):
        print("Neural network initiallizer")
        print("How many inputs do you want")
        self.layerSize.append(int(input("please provide an integer ?")))
        
        print("Create Layers")
        while True:
            print("press (1) to create a layer , press (2) to exist")
            in_ = input()
            if(in_ == '1'):
                
                print("how many nodes inside the the layer")
                
                num_layer = int(input())
                
                print("which activation do you want : 0 - sigmoid , 1 - linear , 2 -tanh , 3- softmax")
                
                activation = int(input())
                while activation > 3 or activation < 0:
                    
                    activation = int(input("please provide a positif integer between 0 and 3"))
                    
                self.AddLayer(num_layer , activation)
            if in_ == "2":
                break
    
    
    
    
    #whenever the base class creates the init file it calls this functions for custom params
    def CustomParams(self):
        activationsfunc = []
        shapes = []
        for layer in self.layers:
            activationsfunc.append(layer.activationType)
            shapes.append(layer.params.shape)
        
        return [[self.PARAMS_ACTIVATION , activationsfunc ],[self.PARAMS_SHAPES , shapes ]]
    
    
    
    
    #whenever the base class wants to create a random parameter file it calls this functions to determine the contents
    def createRandomState(self):
        params = []
        for layer in self.layers:
            params.append(Utilitys.createRandomParams(layer.params.shape).tolist())
       
        return json.dumps(params)
    
    
    
    
    def TurnStringToModel(self , params , model_settings):
        params = json.load(params)
        model_settings = json.load(model_settings)
        
        self.layerSize = [model_settings[self.PARAMS_SHAPES][0][1]]
        for x in range(0 , len(model_settings[self.PARAMS_ACTIVATION])):
            new_layer = layer(3, 3,  model_settings[self.PARAMS_ACTIVATION][x])
            new_layer.params = np.array(params[x])
            self.layerSize.append(new_layer.params.shape[1])
            self.layers.append(new_layer)
        return
    
   
    
    
    
    
    #given our model create a javascript script of our model
    def BakeJavascript(self):
        
        return
    
    def TurnModelToString(self):
        params = []
        for layer in self.layers:
            params.append(layer.params.tolist())
        return json.dumps(params)
    def BakeCsharpModel(self):
        return
        
    #given ou
    def BakeCsharpOptimizationAlgorithm(self):
        return
    
    # adds a layer to the neural network
    def AddLayer(self , numberOfNodes , activationFunctionType):
        self.layers.append(layer(numberOfNodes, self.layerSize[-1] + 1, activationFunctionType))
        self.layerSize.append(numberOfNodes)
    
    
    
    
    
    #given a x and a set of its layers compute the y of the x
    #we want to put the layers as param because we will later be making , microvariations and testing them
    #
    
    
    def ComputeModel(self , inputs , layers):
        inputs = np.append(inputs , 1)
        result = layers[0].computeLayer(inputs)
       
        for layerindex in range( 1 , len(self.layers)):
            result = np.append(result , 1)
            result = layers[layerindex].computeLayer(result)

        return result
    
    
    
    # given a set of x's compute the y's
    def RunEpoch(self , xDataSet , layers):
        yHatDataSet = self.ComputeModel(xDataSet[0] , self.layers)
        
        for x in range(1, len(xDataSet)):
            yHatDataSet = np.vstack((yHatDataSet, self.ComputeModel(xDataSet[x] , layers)))
        return yHatDataSet
    
    
    
    #given a set of x's and y's , compute the y's that the model would have given then compare it to the real thing
    def GetLoss(self , xDataSet , yDataSet , layers):
        yHatDataSet = self.RunEpoch(xDataSet , layers)
        
        loss = Utilitys.compute_loss(yDataSet, yHatDataSet)
        return loss
    
    
    
   
        
    def OptimizeAmount(self , amount , dataX, dataY , OptimizingSteps):
        
        InitalStep = 10
        step = InitalStep
        loss_variation = [100.00]
        new_layers = self.layers
        
        
        MINIMAL_DISTANCE = 0.0001
        
        
        MINIMUM_LOSSVARIATION = 0.00001
        STEP_BUMP = 0.05
        i = 10
        xtest = dataX[i]
        ytest = dataY[i]
        
        
        
        
        for epoch in range(0 , amount):
            
            loss = self.GetLoss(dataX , dataY , new_layers)
            print("cycle # : " + str(epoch))
            print(loss)
            
            
            
            
            if(loss >= loss_variation[-1]):
                new_layers = self.layers 
                if(loss == loss_variation[-1]):
                   step = step * 2
                else:
                   step = step / 2
                print('adjusting training process')
                xtest = dataX[i]
                ytest = dataY[i]
                i+=1
                
            else:
                self.layers = new_layers
                
                if(loss_variation[-1] - loss <  MINIMUM_LOSSVARIATION):
                    step = step + STEP_BUMP
                    xtest = dataX[i]
                    ytest = dataY[i]
                    i+=1
                
                loss_variation.append(loss)
                
            
                
            initalLoss = Utilitys.compute_loss(self.ComputeModel(xtest , new_layers) , ytest)

            for OptimizingStep in range(0 , OptimizingSteps):
                
                for layerindex in range(0, len(self.layers)):
                    
                    for x in range(0 , new_layers[layerindex].params.shape[0] ):
                        for y in range(0 , new_layers[layerindex].params.shape[1] ):
                            
                            layers = new_layers
                            layers[layerindex].params[x , y] += MINIMAL_DISTANCE
                            variation = (Utilitys.compute_loss(self.ComputeModel(xtest , layers) , ytest) - initalLoss) / MINIMAL_DISTANCE
                            
                            new_layers[layerindex].params[x, y] -= variation * step
                            #apply over each node ...
                          
                
                                
            
        Utilitys.baseUtulitys.SaveModel(self)
            
            
            
            
            
            
        



