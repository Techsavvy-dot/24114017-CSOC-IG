import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time

df = pd.read_csv(r'/kaggle/input/housing-dataset/housing.csv')
df.head()

df.shape

df.isnull().sum()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["ocean_proximity"] = label_encoder.fit_transform(df["ocean_proximity"])
df

plt.scatter(df['total_bedrooms'],df['median_house_value'])
plt.xlabel("total bedrooms")
plt.ylabel("median house value")
plt.show()

import seaborn as sns
corr_matrix = df.corr()
corr_with_median_value = corr_matrix["median_house_value"].sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Housing Dataset')
plt.show()

print("Correlations with median_house_value:\n", corr_with_median_value)

df = df.dropna(axis=0)

df.isnull().sum()

x=df['median_income']
y = df["median_house_value"]
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)

# [markdown]
#  part 1 : using purely python for multivariable linear regression

def dot_product(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result
    
def scalar_multiply(scalar, vector):
    return [scalar * x for x in vector]
    
def vector_subtract(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]
    
def vector_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]


class multivar_lin_reg:
    def _init_(self):
        self.bias = None
        self.weight = None

    def fit(self,x,y):# y is the dependent variable. It depends on x1,x2 and x3 which are independent variable.
        x_mean = x.mean()
        y_mean = y.mean()
    
        num=0
        den=0
        for i in range(len(x)):
            num += ((x.iloc[i]- x_mean)*(y.iloc[i] - y_mean))
            den += (x.iloc[i] - x_mean)**2
        self.weight = num / den 
        self.bias = 0
    
        
        lr = 0.001 #learning rate
        iterations = 2000
        
        
        start_time = time.time()
        mse_list = []
        
        for iteration in range(iterations):
            prediction = self.bias + self.weight*x
            errors = prediction - y
            
            mse = (errors**2).mean()
            mse_list.append(mse)
            m = len(y)
            gradient = (2/m)*dot_product(list(x.T),list(errors))
            gradient_b = (2/m) * sum(errors)
            self.weight -= lr * gradient
            self.bias -= lr*gradient_b
        end_time = time.time()
        print(f'converging time:{end_time - start_time}')
        print(self.weight)
        print(self.bias)
        plt.figure(figsize=(11,7))
        plt.plot(range(iterations),mse_list)
        plt.title("cost function")
        plt.xlabel("iterations")
        plt.ylabel("mean square error")
        plt.show()
        
    

    def predict(self, x):
        x_mean = x.mean
        predictions = []
        for i in range(x.shape[0]):
            prediction = self.bias + self.weight*x.iloc[i]
            predictions.append(prediction)
        return predictions  



model= multivar_lin_reg()
model.fit(x_train,y_train)


# [markdown]
# The loss function measures how far predictions are from actual values:
#
# 𝑀𝑆𝐸=  (1/𝑚)∑(predicted value −actual value)^2
#
#
# where 𝑚 is the number of training samples.
#
# Gradient of Weight :
#
# ∂𝑀𝑆𝐸/∂𝑤  =  −2/𝑚∑𝑥(predicted value−actual value)
#
# Gradient of Bias :
#
# ∂𝑀𝑆𝐸/∂𝑏  =  −2/𝑚∑(predicted value−actual value)

y_pred = model.predict(x_test)

errors = y_pred - y_test
errors

mae_1 = abs(errors).mean()
mae_1

t=0
for i in errors:
    t += i**2
rmse_1 = (t/len(errors))**0.5
rmse_1

mean_y = sum(y_test) / len(y_test)
den=0
for i in range(len(y_test)):
    den += (y_test.iloc[i]-mean_y)**2
num=0
for i in range(len(y_test)):
    num += (y_test.iloc[i]-y_pred[i])**2

r2_1 = 1-(num / den) 
r2_1

# [markdown]
#  Part 2: Optimized Numpy Implementation

class numpy_multivar_lin_reg:
    def __init__(self):
        self.bias = None
        self.weight = None

    def fit(self, x, y):
        x = np.array(x)  # Convert to NumPy array
        y = np.array(y)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
    
        # Compute initial weight using least squares
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean)**2)
        self.weight = num / den
        self.bias = 0  # Initializing bias
        
        # Gradient Descent parameters
        lr = 0.001  # Learning rate
        iterations = 2000
        
        start_time = time.time()
        mse_list = []

        for iteration in range(iterations):
            prediction = self.bias + self.weight * x
            errors = prediction - y
            
            # Compute Mean Squared Error
            mse = np.mean(errors**2)
            mse_list.append(mse)
            
            m = len(y)
            gradient_w = (2/m) * np.dot(x.T, errors)  # Weight gradient
            gradient_b = (2/m) * np.sum(errors)  # Bias gradient
            
            # Update parameters
            self.weight -= lr * gradient_w
            self.bias -= lr * gradient_b
        
        end_time = time.time()
        print(f'Converging time: {end_time - start_time:.6f} seconds')
        print(f'Final weight: {self.weight:.6f}')
        print(f'Final bias: {self.bias:.6f}')
        
        # Plot cost function convergence
        plt.figure(figsize=(11, 7))
        plt.plot(range(iterations), mse_list)
        plt.title("Cost Function Convergence")
        plt.xlabel("Iterations")
        plt.ylabel("Mean Squared Error")
        plt.show()

    def predict(self, x):
        x = np.array(x)  # Convert input to NumPy array
        return self.bias + self.weight * x

numpy_model = numpy_multivar_lin_reg()
numpy_model.fit(x_train,y_train)

y_pred = model.predict(x_test)
errors = y_pred - y_test
errors

mae_2 = np.mean(np.abs(errors))
print(f"mean absolute error is {mae_2:.4f}")

rmse_2 = np.sqrt(np.mean(errors**2))
print(f"root mean square error is {rmse_2:.4f}")

ss_total = np.sum((y - np.mean(y))**2)
ss_residual = np.sum(errors**2)
r2_score_2 = 1 - (ss_residual / ss_total)
print(f"R2_score is {r2_score_2:.4f}")

# [markdown]
#  Part 3 :Using Scikit-learn library

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

model = LinearRegression()

x_train = x_train.values.reshape(-1, 1) 

x_test = x_test.values.reshape(-1, 1) 

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# Calculate the fitting duration
fitting_duration = end_time - start_time
print(f'Fitting Duration: {fitting_duration:.4f} seconds')

predictions = model.predict(x_test)
predictions

mse_3 = mean_squared_error(y_test, predictions)
mae_3 = mean_absolute_error(y_test, predictions)
rmse_3 = np.sqrt(mse_3)
r2_3 = r2_score(y_test, predictions)

print(f'mean absolute error: {mae_3:.4f}')
print(f'Root Mean Squared Error: {rmse_3:.4f}')
print(f'R-squared Score: {r2_3:.4f}')

metrics = ['pure python','Numpy','Scikit-learn library']
values = [rmse_1,rmse_2,rmse_3]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'orange','red'])
plt.ylabel('Value')
plt.title('Root Mean Squared Error')
plt.show()

metrics = ['pure python','Numpy','Scikit-learn library']
values = [mae_1,mae_2,mae_3]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'orange','red'])
plt.ylabel('Value')
plt.title('Mean Absolute Error')
plt.show()

metrics = ['pure python','Numpy','Scikit-learn library']
values = [r2_1,r2_score_2,r2_3]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'orange','red'])
plt.ylabel('Value')
plt.title('R2_score')
plt.show()

