#importing lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#reading data into pandadataframe nl
nl = pd.read_csv("M:\shan\Assignment Code naved Copy/ce889_dataCollection.csv")
nan_value = np.nan
nl.replace("", nan_value, inplace=True) #subsituiting the empty string with NaN
nl.dropna(inplace=True)#droping NaN values
nl.drop_duplicates(keep="first", inplace=True) #dropping duplicates values

column_title = ["X_first", "Y_first", "X_Second", "Y_Second"] #assigning column names 
nl.columns = column_title #nl columns names are column tile which we assigned
#here we start partitionary process
#defining two sets X and y labels from dataframe nl
X = nl[["X_first", "Y_first"]]
y = nl[["X_Second", "Y_Second"]]
#we use here Minmaxscaler to scale both x and y to normalise between range 0-1
scaler = MinMaxScaler()
X_scaled, y_scaled = scaler.fit_transform(X), scaler.fit_transform(y) #here we are fitting x__scaled and y_scaled
#spliting data 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.33, random_state=42)
#plotting graph
plt.figure(figsize=(10, 4)) #resolution of sreen size
plt.subplot(141)
sns.histplot(nl[["X_first", "Y_first"]], kde=True) #first 2 coloumns
plt.show()
#creating a neural network taking input hidden layer and output
class Neural_Network_Class:
    def __init__(self, layers=None, learning_rate=None, epochs=None):
        self.layers = layers or (2, 3, 2) #input hidden layer and output
        self.learning = learning_rate or 0.01
        self.epochs = epochs or 1000
        self.params = {f"W{j}": np.random.randn(self.layers[j - 1], self.layers[j]) for j in range(1, len(self.layers))}
        self.params.update({f"b{i}": np.zeros((1, self.layers[i])) for i in range(1, len(self.layers))})
        self.losses = []
#sigmoid activation
    def sigmoid(self, N):
        return 1 / (1 + np.exp(-N))
#derivating sigmoid function train the nn like gradient descent 
#here we are calculating derivative of the sigmoid
    def derivative_sigmoid(self, N):
        return N * (1 - N)

    def eta(self, x):
        ETA = 0.0000000001
        return np.clip(x, a_min=ETA, a_max=None)

    def regression_RMSE(self, y_pred, y):
        return np.sqrt(np.mean((y_pred - y) ** 2))
#here we start size comparison  and weights 
    def initialize_weights(self):
        np.random.seed(2) #getting random values
        #weight matrix defining 
        self.params["W1"] = np.random.rand(self.layers[0], self.layers[1])#w1 joins the input layer and hdlayer (2,3)
        #size of bias for hd layer 
        self.params["b1"] = np.random.randn(self.layers[1],)
        self.params["W2"] = np.random.rand(self.layers[1], self.layers[2])
        self.params["b2"] = np.random.randn(self.layers[2])

    def feed_forward_propagation(self):
        input_layer = np.dot(self.X, self.params["W1"]) + self.params["b1"] 
        hidden_layer = self.sigmoid(input_layer)
        output_layer = np.dot(hidden_layer, self.params["W2"]) + self.params["b2"]

        # saving hidden for backpropagation
        self.params["hidden_layer"] = hidden_layer
        #loss function for feed forward propagation
        loss = np.sqrt(np.mean((self.y - output_layer) ** 2))
        #returing the output layer and loss value for backpropagation updating weight
        return output_layer, loss

    def nn_backpropagation(self, y_pred):
        del_output = (self.y - y_pred) * self.derivative_sigmoid(y_pred) 
        del_hidden = del_output.dot(self.params["W2"].T) * self.derivative_sigmoid(self.params["hidden_layer"])

        self.params["W1"] += self.learning * self.X.T.dot(del_hidden)
        self.params["b1"] += self.learning * np.sum(del_hidden, axis=0)
        self.params["W2"] += self.learning * self.params["hidden_layer"].T.dot(del_output)
        self.params["b2"] += self.learning * np.sum(del_output, axis=0)
#learining the pattern from data which we are training 
    def fit(self, X, y):
        self.X, self.y = X, y
        self.initialize_weights()
        self.train()

    def train(self):
        for i in range(self.epochs):
            output_layer, loss = self.feed_forward_propagation()
            self.nn_backpropagation(output_layer)
            self.losses.append(loss)

    def make_prediction(self, input_data):
        hidden_layer = self.sigmoid(np.dot(input_data, self.params["W1"]) + self.params["b1"])
        return self.sigmoid(np.dot(hidden_layer, self.params["W2"]) + self.params["b2"])
    
    def cal_rsme(self, y, y_pred):
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse
#here we are updating the weights
    def nn_updated_weights(self, filename):
         nn_updated_weights = {
        "W1": self.params["W1"].tolist(),
        "W2": self.params["W2"].tolist()
    }
         np.savez(filename, **nn_updated_weights)

#plotting rsme 
    def plot(self):
        plt.plot(self.losses)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.show()

#assigning the neurak network class as nn_neural_network
nn_neural_network = Neural_Network_Class()
#telling the training data to learn the pattern 
nn_neural_network.fit(X_train, y_train)
#plotting the xtrain and ytrain data
nn_neural_network.plot()
#creating a npz file for updated weight
nn_neural_network.nn_updated_weights('file_name.npz')