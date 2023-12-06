

import numpy as np
import math
import keras


DATASET_PATH = "C:/Users/sukho/OneDrive/Documents/DataSets/mnist/mnist.npz"

def LoadDataSetMNIST():
    return keras.datasets.mnist.load_data(path = DATASET_PATH)

def LoadTrainSet():
    dataset =  keras.datasets.mnist.load_data(path = DATASET_PATH)
    amount = 1000
    return dataset[0][0][0:amount] , dataset[0][1][0:amount]

def TestTrainSet():
    return 1
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def linear(x):
    return x

def relu(x):
    return max(0 , x)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
    # remove largest value
    expx = np.exp(x - np.max(x))
    return expx / expx.sum(axis=0, keepdims=True)



    
activationFunctions = [sigmoid , linear , tanh , softmax]

def compute_loss(y , realY):
    
    lenght = realY.shape[0]
    return (1/lenght) * np.sum((y - realY) **2)

def createzeros(shape):
    return np.zeros(shape)
   
def createRandomParams(size ):
    return np.random.random(size)





class layer:
    def __init__(self , numberOfNodes , paramSize , activationFunction):
        
       self.params = createRandomParams([numberOfNodes , paramSize])
       self.activationType = activationFunction
       
    def computeLayer(self , inputs):
        
        result = np.matmul( self.params , inputs)
        
        return activationFunctions[self.activationType](result)
  
    
  
    
  
    
  
class NeuralNetwork:
    def __init__(self , numberofInputs):
        self.layerSize = [numberofInputs]
        self.layers = []
    
    
    #given our model create a javascript script of our model
    def BakeJavascript(self):
        return
    
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
    def RunEpoch(self , xDataSet):
        yHatDataSet = self.ComputeModel(xDataSet[0] , self.layers)
        
        for x in range(1, len(xDataSet)):
            yHatDataSet = np.vstack((yHatDataSet, self.ComputeModel(xDataSet[x] , self.layers)))
        return yHatDataSet
    
    
    
    #given a set of x's and y's , compute the y's that the model would have given then compare it to the real thing
    def GetLoss(self , xDataSet , yDataSet):
        yHatDataSet = self.RunEpoch(xDataSet)
        print(yHatDataSet)
        print(yHatDataSet.shape)
        loss = compute_loss(yDataSet, yHatDataSet)
        return loss
    
    
    
    #given a x and a y , create a matrix that shows the variation of computed y and real y if we make micro adjustement to each variable of the network
    def ComputeDerivativeGradient(self, xtest , ytest):
         layers = self.layers
         
         MINIMALDISTANCE = 0.0001
         
         gradient = []
         #here we get the initial loss
         initalLoss = compute_loss(self.ComputeModel(xtest , layers) , ytest)
         
         #we fill the gradient with zeros
         for layer in self.layers:
             gradient.append(createzeros(layer.params.shape))
        
         #then for each param we use the derivative formula (f(x + h) - f(x)) / h
         #then so we get the derivate along each parameter
         
         for layerindex in range(0, len(self.layers)):
             
             for x in range(0 , layers[layerindex].params.shape[0] ):
                 for y in range(0 , layers[layerindex].params.shape[1] ):
                     
                     layers = self.layers
                     layers[layerindex].params[x , y] += MINIMALDISTANCE
                     gradient[layerindex][x , y] = (compute_loss(self.ComputeModel(xtest , layers) , ytest) - initalLoss) / MINIMALDISTANCE
         
         #and we return it
         return gradient
       
         
         
    def runBackPropagation(self , xDataSet , yDataSet , step):
        
        loss = self.GetLoss(xDataSet , yDataSet)
        print("loss amount : " + str(loss))
        gradient = self.ComputeDerivativeGradient(xDataSet[0] , yDataSet[0])
        
        #we have our average loss and a derivative of each variable
        #now we want to find whitch derivative is the closet to our loss and adjust the real value of that one
        lowest_layer_ind =0
        lowest_x = 0
        lowest_y = 0
        print(gradient[lowest_layer_ind][0 , 0])
        lowest_distance = abs(loss -  gradient[lowest_layer_ind][lowest_x , lowest_y])
        for layerindex in range(0, len(self.layers)):
            for x in range(0 , self.layers[layerindex].params.shape[0]):
                for y in range(0 , self.layers[layerindex].params.shape[1]):
                    distance = abs(loss - gradient[layerindex][x,y])
                    if( distance < lowest_distance):
                        lowest_distance = distance
                        lowest_layer_ind = layerindex
                        lowest_x = x
                        lowest_y = y
        self.layers[lowest_layer_ind].params[lowest_x , lowest_y] -= step
        
        
    def OptimizeAmount(self , amount , dataX, dataY):
        for epoch in range(0 , amount):
            self.runBackPropagation(dataX, dataY, 0.1)
            
        
        
    def SaveModel(self , path):
        return
    def LoadModel(self , path):
        return               

def giveArrayStructure(array):
    return

def Format(x):
    new_x = []
    for image in x:
        new_list = np.array([])
        for row in image:
            new_list = np.concatenate((new_list , row) )
        new_x.append(new_list)
    return new_x

def ForamtYMNIST(y):
    
    entrie = createzeros(10)
    entrie[y[0]] = 1 
    new_y = np.array([entrie])
    
    for num in range(1 , len(y)):
        entrie = createzeros(10)
        entrie[y[num]] = 1
        new_y = np.vstack((new_y , entrie))
    
    return new_y



x , y = LoadTrainSet()

y = ForamtYMNIST(y)
x = Format(x)


learning_model = NeuralNetwork(len(x[0]))

learning_model.AddLayer(128, 0)
learning_model.AddLayer(10 , 3)


learning_model.OptimizeAmount(10 , x, y)

