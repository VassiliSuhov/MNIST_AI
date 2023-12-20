

import numpy as np
import math
from . import Utilitys
import random
import pyperclip
import json
#your path here 




class layer:
    def __init__(self , numberOfNodes , paramSize , activationFunction):
        
       self.params = Utilitys.createRandomParams([numberOfNodes , paramSize])
       self.activationType = activationFunction
       
    def computeLayer(self , inputs):
        
        result = np.matmul( self.params , inputs)
        
        return Utilitys.activationFunctions[self.activationType](result)
    
    
    
  
    

  
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
                
                for num in range(0 , len(Utilitys.activationNames)):
                    print(str(num) + "  " + Utilitys.activationNames[num])
                
                activation = int(input())
                while activation > len(Utilitys.activationNames) or activation < 0:
                    
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
        cpt = 0
        for layer in self.layers:
            for x in range(0,layer.params.shape[0]):
                pyperclip.copy("neuron" + str(cpt) + " = " + str(layer.params[x].tolist()))
                cpt+=1
                i = input()
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
    
    
    def ComputeGradients(self , x ,y):
        
        data_layers = self.FullDataCompute(self.layers, x)
        len_data_layers = len(data_layers) -1 
        
        next_values = Utilitys.BackpropagateLossFunction(data_layers[-1], y )
        gradients  = []
        start_ = len(self.layers) -1
        
        for layerIndex in range(0 , len(self.layers)):
            
            
            next_values = Utilitys.backPropFunctions[self.layers[start_].activationType](next_values , data_layers[len_data_layers],data_layers[len_data_layers - 1] )
            len_data_layers -= 2
            results = Utilitys.CalculateLayerDerivative(next_values, self.layers[start_].params,np.append( data_layers[len_data_layers] , 1))
            next_values = results[0]
            gradients.insert(0, results[1])
            start_ -=1
            
        """  
        step_1 = Utilitys.BackPropSigmoid(error , data_layers[-1] , None)
         
        
        
        
        #step 2 we have the gradients of al the sigmoid results  now we get according to the layers
        results  = Utilitys.CalculateLayerDerivative(step_1 , self.layers[1].params , np.append(data_layers[-3] , 1))
        inputs_ = results[0]
        layer_2 = results[1]
        
        
       
        
        softmax_derivate = Utilitys.BackPropSoftmax(inputs_,  data_layers[-3],  data_layers[-4])
        
        results = Utilitys.CalculateLayerDerivative(softmax_derivate, self.layers[0].params , np.append(x , 1))
        
        layer_1 = results[1]
        
        
        gradients = [layer_1 , layer_2]
        """
        return gradients
    
    
    def BackPropagation(self , x , y , step , gradient , gradient_size):
        
        
        new_layers = self.layers
        gradients = gradient
        
        for layerindex in range(0 , len(gradients)):
            for x_ in range(0, gradients[layerindex].shape[0]):
                for y_ in range(0, gradients[layerindex].shape[1]):
                    new_layers[layerindex].params[x_, y_] -=  (gradients[layerindex][x_, y_] * step)/gradient_size
        
        
      
        return new_layers
        #now we get the layers
        #backwards the 
        #step_3 = Utilitys.BackPropSoftmax(inputs_, data_layers[-3] , None)
        
        #inputs_1 , layer_1 = Utilitys.CalculateLayerDerivative(step_3 ,  self.layers[0].params, self.data_layers[0])
        
        #print(layer_1)
        
    ##instead of just computing the result , it saves the calculations after each step
    
    def FullDataCompute(self, layers , x ):
        
        data_layers = [x]
        
        for layerIndex in range(0, len(layers)):
            
            x = np.append( data_layers[-1], 1)
            data_layers.append( np.matmul(layers[layerIndex].params , x ))
            data_layers.append(Utilitys.activationFunctions[layers[layerIndex].activationType](data_layers[-1]))
            
        return data_layers
                              
        
        
    
    def OptimizeAmount(self , amount , dataX, dataY , OptimizingSteps,sample_amount):
        data_length = len(dataX) - 1 
        
        loss = self.GetLoss(dataX, dataY, self.layers)
        print("global error")
        print(loss)
        
        for epoch in range(0 , amount):
            index = random.randint(0, data_length)
            gradient = self.ComputeGradients(dataX[index], dataY[index])
            
            
            for z in range(0 , sample_amount):
                index = random.randint(0, data_length)
                new_grad = self.ComputeGradients(dataX[index], dataY[index])
                for layer in range(0 , len(gradient)):
                    gradient[layer] += new_grad[layer]
              
                
            new_layer = self.BackPropagation(dataX[index], dataY[index], OptimizingSteps , gradient , sample_amount + 1)
            
            new_loss = self.GetLoss(dataX, dataY, new_layer)
            
            print("global loss")
            print(new_loss)
            
            if(new_loss < loss):
                print("updating loss")
                loss = new_loss
                self.layers = new_layer
            
                
                                
            
        Utilitys.baseUtulitys.SaveModel(self)
            
            
            
            
            
            
        



