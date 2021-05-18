
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import  log_loss
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import log, dot, e
from numpy.random import rand

class LogisticRegression:
    def accuracy(self,actual, predicted):
	    correct = 0
	    for i in range(len(actual)):
		    if actual[i] == predicted[i]:
			    correct += 1
	    return correct / float(len(actual)) * 100.0
    
    def sigmoid(self, z): return 1 / (1 + e**(-z))
    
    def cost_function(self, X, y, weights):                 
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)
    
    def fit(self, X, y, epochs=25, lr=0.05):        
        loss = []
        weights = rand(X.shape[1])
        
        N = len(X)
                 
        for _ in range(epochs):        
            # Gradient Descent
            y_hat = self.sigmoid(dot(X, weights))
            weights -= lr * dot(X.T,  y_hat - y) / N            
            # Saving Progress
            loss.append(self.cost_function(X, y, weights)) 
            print('epoch '+ str(_))
            print(self.cost_function(X, y, weights))

        self.weights = weights
        self.loss = loss
    
    
    def predict(self, X):        
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]


data = pd.read_csv('/home/aadeesh/diabetes2.csv')
df = pd.DataFrame(data)
X= df[df.columns[[1,2,3,4,5,6,7,8]]].values
y= df[df.columns[-1]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# Create linear regression object
#model = linear_model.LogisticRegression(max_iter=10000,solver='saga')
model = LogisticRegression()


# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)


