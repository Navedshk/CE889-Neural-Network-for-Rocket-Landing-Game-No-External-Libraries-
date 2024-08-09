#importing from navedaltered and class neural network class  
from navedaltered import Neural_Network_Class  
import numpy as np

class NeuralNetHolder: 

    def __init__(self):
        super().__init__()
        
        #here we are calling nnc
        self.neuralnet = Neural_Network_Class()   
        #function load and to create file with file.name.npz
        
    def load(self,f):
        return self.neuralnet.defining_the_weights('file_name.npz')  
    #here we are prediction and taking 2 parameters 1 self and 2 input_row from navedaltered
    def predict(self, input_row):
        #writing code for processed input and x and y 
        #taking the value from input_row to split it in the row
        pattern= [float(i) for i in input_row.split(",")] 
        #here we are calculating x with row 
        x = pattern[0]/100 * 10  
        #here we are calculating y with row
        y = pattern[1]/100* 10 
        print(x,y)
        #here we are returning the x and y   
        return x,y        


