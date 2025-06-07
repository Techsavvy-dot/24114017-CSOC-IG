import numpy as np 
import pandas as pd
!pip install gensim

train_df = pd.read_csv(r'/kaggle/input/train-csv/train.csv')

test_df = pd.read_csv('/kaggle/input/train-csv/train.csv')

train_df = train_df.dropna()

train_df.iloc[:,0].value_counts()

# [markdown]
#  RNN 

sentences = [review.split() for review in train_df.iloc[:,2]]

from gensim.models import KeyedVectors,Word2Vec

model = Word2Vec(sentences, vector_size=16, window=3, min_count=100000, workers=1000) 
model.wv.save_word2vec_format("got_word2vec.txt", binary=False)

word_vectors = KeyedVectors.load_word2vec_format("got_word2vec.txt", binary=False)

def words_to_vector(text, model, vector_size=16):
    
    words = text.split()  # Tokenize text
    vectors = [model[w] for w in words if w in model]  # Convert words to embeddings
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

x = [words_to_vector(review,word_vectors,16) for review in train_df.iloc[:,2]]
y = train_df.iloc[:,0]
x = np.array(x)
y = np.array(y)
x = x.reshape((x.shape[0], 1, x.shape[1]))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)
y_tensor = (y_tensor.view(-1, 1)).float()
y_tensor = y_tensor - 1
import gc
del x, y
gc.collect()

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first = True,nonlinearity='tanh')
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
            )
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)  
        out = out[:, -1, :]  
        out = self.fc(out) 
        return out

input_size = 16
hidden_size = 64
num_layers = 2
num_classes = 2
learning_rate = 0.01

model = RNN(input_size,hidden_size,num_layers,num_classes).to(device)
del RNN,input_size,hidden_size,num_layers,
gc.collect()
criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
y_tensor = y_tensor.view(-1, 1)
for epoch in range(100):
    optimizer.zero_grad()  
    outputs = model(x_tensor) 
    loss = criterion(outputs, y_tensor)  
    loss.backward()  
    optimizer.step() 
    del outputs
    gc.collect()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')

del criterion,optimizer,train_df,sentences
gc.collect()

x_test = np.array([words_to_vector(review,word_vectors,16) for review in test_df.iloc[:,2]])
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
del x_test,word_vectors,words_to_vector
predictions = model(x_test_tensor)
del model
gc.collect()

from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
y_test = np.array(test_df.iloc[:,0])
y_test = torch.tensor(y_test, dtype=torch.long).to(device)
y_test = (y_test.view(-1, 1)).float()

accuracy = accuracy_score(y_test, predictions.cpu().numpy())
print(f'Accuracy: {accuracy:.4f}')
f1 = f1_score(y_test, predictions.cpu().numpy(), average="binary") 
print(f'F1-Score: {f1:.4f}')
conf_matrix = confusion_matrix(y_test, predictions.cpu().numpy())
print("Confusion Matrix:")
print(conf_matrix)

# [markdown]
#  LSTM

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Replace RNN with LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer with sigmoid for binary classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]  # Take last output in sequence
        out = self.fc(out)  # Apply classification layer
        return out

# Hyperparameters
input_size = 16
hidden_size = 64
num_layers = 2
num_classes = 2
learning_rate = 0.01

# Initialize model
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(100):
    optimizer.zero_grad()  
    outputs = model(x_tensor)  # Remove `torch.no_grad()` (needed for training)
    loss = criterion(outputs, y_tensor)  
    loss.backward()  
    optimizer.step() 

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')

# Clean up memory
del x_tensor, y_tensor
gc.collect()

predictions = model(x_test_tensor)

accuracy = accuracy_score(y_test, predictions.cpu().numpy())
print(f'Accuracy: {accuracy:.4f}')
f1 = f1_score(y_test, predictions.cpu().numpy(), average="binary") 
print(f'F1-Score: {f1:.4f}')
conf_matrix = confusion_matrix(y_test, predictions.cpu().numpy())
print("Confusion Matrix:")
print(conf_matrix)

