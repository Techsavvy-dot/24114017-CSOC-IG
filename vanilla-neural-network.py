import numpy as np # linear algebra
import pandas as pd
from numpy.random import randn


df = pd.read_csv(r'/kaggle/input/medical/KaggleV2-May-2016.csv')

df.isnull().sum()

df.shape

pd.set_option('future.no_silent_downcasting', True)
df.replace({'Gender': {'M':0,'F':1}, 'No-show': {'Yes':1,'No':0}}, inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt
corr = df[['Gender','Age','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received','No-show']].corr()
corr_with_noshow = corr["No-show"].sort_values(ascending=False)
plt.figure(figsize=(18,9))
sns.heatmap(corr,annot=True,cmap='coolwarm', fmt=".2f")
plt.show()
print(corr_with_noshow)

x= df[['Age','Hipertension','Diabetes','Handcap','Gender','Scholarship','SMS_received']]
y=df['No-show']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

y_train = y_train.astype(int)
x_train = x_train.astype(int)
y_test = y_test.astype(int)
x_test = x_test.astype(int)

# [markdown]
#  Part 1- Neural Network from scratch


class ANN:
    def __init__(self, input_size=7, hidden_size=50):
        # Initialize weights and biases
        self.input_weights = np.random.uniform(-0.5, 0.5, size=(input_size, hidden_size)).astype(np.float32)
        self.output_weights = np.random.uniform(-0.5, 0.5, size=(hidden_size, 1)).astype(np.float32)
        self.b1 = np.random.uniform(-0.5, 0.5, size=hidden_size).astype(np.float32)
        self.b2 = np.random.uniform(-0.5, 0.5, size=1).astype(np.float32)

    def ReLU(self,z):
        return np.maximum(0,z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x, y): 

        # Hidden layer activations
        z = x @ self.input_weights + self.b1
        h = self.ReLU(z)
        # Output layer activation
        net_out = h @ self.output_weights + self.b2
        out = self.sigmoid(net_out)
        return out,h
        
        
    

    def backward(self, x, y, lr=0.01): 
        out,h = self.forward(x,y) 
        y = y.astype(np.float32) 
        out = out.astype(np.float32) 
        
        # Gradients for output layer 
        d_out=np.zeros_like(out)
        d_out = (y.to_numpy().reshape(out.shape) - out) * out * (1 - out)
        grad_output_weights = h.T @ d_out 
        grad_b2 = np.sum(d_out, axis=0) 
       
        # Gradients for hidden layer 
        d_h = (d_out @ self.output_weights.T) * h * (1 - h) 
        grad_input_weights = x.T @ d_h 
        grad_b1 = np.sum(d_h, axis=0) 
        
        # Update weights and biases 
        self.output_weights += lr * grad_output_weights 
        self.b2 += lr * grad_b2 
        self.input_weights += lr * grad_input_weights 
        self.b1 += lr * grad_b1
        
    def predict(self, x):
         
        # Hidden layer activations
        z = x @ self.input_weights + self.b1
        h = self.ReLU(z)
        # Output layer activation
        net_out = h @ self.output_weights + self.b2
        out = self.sigmoid(net_out)
        out = (out > 0.5).astype(int)
        return out


import time

start_time = time.time()
ann_model = ANN()
ann_model.forward(x_train,y_train)
for i in range(500):
    ann_model.backward(x_train, y_train)
end_time = time.time()
print(f'convergence time:{end_time-start_time}')

y_pred = ann_model.predict(x_test)


y_pred,y_test

y_pred = y_pred.astype(int)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,precision_score, recall_score, auc

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

#PR-AUC
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
pr_auc = auc((recall,1),(precision,1))
print("pr_auc:\n", pr_auc)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# [markdown]
#  Part 2- Using Pytorch

import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # Hidden Layer
        self.layer2 = nn.Linear(hidden_size, output_size)  # Output Layer

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.layer1(x))  # Hidden layer activation
        x = self.layer2(x)  # Output layer WITHOUT sigmoid
        return x


input_size = x_train.shape[1]
output_size = 1
hidden_size = 50

model = SimpleNN(input_size, hidden_size, output_size)

# Define Loss function and Optimizer

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)   # Stochastic Gradient Descent

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x_train.values, dtype=torch.float32)  # Convert DataFrame to Tensor
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_tensor = y_tensor.view(-1, 1) 
x_tensor_test = torch.tensor(x_test.values, dtype=torch.float32)  
y_tensor_test = torch.tensor(y_test.values, dtype=torch.float32)



# Training loop
start_time = time.time()
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear previous gradients
    output = model(x_tensor)  # Forward pass
    loss = criterion(output, y_tensor)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights    

    if epoch % 100 == 0:  # Print progress every 100 epochs
        print(f"Epoch {epoch}, Loss: {loss.item()}")

end_time = time.time()
print(f'convergence time:{end_time-start_time}')

# Making predictions
predictions = model(x_tensor_test).detach().numpy()
print("Predictions:", predictions)
pytorch_y_pred = (predictions >= 0.5).astype(int)
print("Thresholded Predictions:", (predictions >= 0.5).astype(int))

# Accuracy
acc = accuracy_score(y_test, pytorch_y_pred)
print("Accuracy:", acc)

# F1-score
f1 = f1_score(y_test, pytorch_y_pred)
print("F1-score:", f1)

#PR-AUC
precision = precision_score(y_test, pytorch_y_pred)
recall = recall_score(y_test,pytorch_y_pred)
pr_auc = auc((recall,1),(precision,1))
print("pr_auc:\n", pr_auc)


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, pytorch_y_pred)
print("Confusion Matrix:\n", conf_matrix)

