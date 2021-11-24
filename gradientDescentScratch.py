import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#The weight and intercept variables are initialized; length in the passed in parameter
def initialize_paramenters(lenw):
    w = np.random.randn(1, lenw)
    b = 0
    return w, b

#Forward propogation returning the z variable, using X (train/test) set and the weights and offset (intercept)
def forward_prop(X,w,b):
    z = w.dot(X) + b
    return z

#Cost function used to calculate the weights//used for GD
def cost_function(z,y):
    m = y.shape[1]
    J = (1/(2*m))*np.sum(np.square(z-y))
    print('MSE: ', J)
    return J

#Back-propogation for the slope
def back_prop(X,y,z):
    m = y.shape[1]
    dz = (1/m)*(z-y)
    dw = np.dot(dz,X.T)
    db = np.sum(dz)
    return dw, db

#Finding new values for w and b based on learning rate
def gradient_descent_step(w, b, dw, db, learning_rate):
    w = w - learning_rate*dw
    b = b - learning_rate*db
    return w, b

def linear_regression_model(X_train, y_train, X_test, y_test, learning_rate, epochs):
    lenw = X_train.shape[0]
    w, b = initialize_paramenters(lenw)

    #Lists the training cost(s)
    costs_train = []
    costs_test = []
    m_train = y_train.shape[1]
    m_test  = y_test.shape[1]

    for i in range (1,epochs+1):
        z_train = forward_prop(X_train,w,b)
        cost_train = cost_function(z_train, y_train)
        dw,db = back_prop(X_train, y_train, z_train)
        w,b = gradient_descent_step(w, b, dw, db, learning_rate)

        if i%10==0:                             #Multiples of 10, for better readability
            costs_train.append(cost_train)      #Stores the cost after each iteration

        #MAE Train (Mean Absolute Error for Train Set)
        MAE_train = (1/m_train)*np.sum(np.abs(z_train-y_train))

        #Training costs and the MAE for test
        z_test = forward_prop(X_test,w,b)
        cost_test = cost_function(z_test,y_test)
        if i%10==0:                             #Multiples of 10, for better readability
            costs_test.append(cost_test)       #Stores the cost after each iteration

        ##MAE Test (Mean Absolute Error for Test Set)
        MAE_test = (1/m_test)*np.sum(np.abs(z_test-y_test))

        #print('Training Cost:', cost_train)
        print('Mean Absolute Error for Train: ', MAE_train)
        print ('Mean Absolute Error for Test: ', MAE_test)
    #Plot for the iterations vs training cost
    plt.plot(costs_train)
    plt.title("Iterations vs Training Cost for Train Set")
    plt.xlabel('Iterations')
    plt.ylabel('Training Cost')
    plt.show()

    #Test costs
    plt.plot(costs_test)
    plt.title("Iterations vs Training Cost for Test Set")
    plt.xlabel('Iterations')
    plt.ylabel('Training Cost')
    plt.show()

    #Calculating r^2
    total_sum_of_difference    = 0
    residual_sum_of_difference = 0
    m_value = y_test.shape[1]
    #use the z_test from about for z_value
    mean_y_test = np.mean(y_test)
    total_sum_of_difference    = sum((y_test[0]-mean_y_test)**2)
    residual_sum_of_difference = sum((y_test[0]-z_test[0])**2)
    r_squared = 1 - (residual_sum_of_difference/total_sum_of_difference)
    print('r^2: ', r_squared)

#Loading the dataset
csv_file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'
raw_data = pd.read_csv(csv_file)
#print(raw_data.head(10))
#print(raw_data.shape)
df = pd.DataFrame(raw_data)
df.drop(columns = ['adviser','32/60'], inplace=True)
#print(df)
#Casting to int to make it compatible for math operations
df = df.astype(int)

plt.scatter(df['198'], df['199'], c = "red", alpha = .5, marker = "o")
plt.title("Published Performance vs Relative Performance")
plt.xlabel("Published Performance")
plt.ylabel("Relative Performance")
plt.show()

plt.scatter(df['6000'], df['199'], c = "red", alpha = .5, marker = "o")
plt.title("Published Performance vs Relative Performance")
plt.xlabel("Cache Memory")
plt.ylabel("Relative Performance")
plt.show()

d = np.random.laplace(loc=15, scale=3, size=500)
n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('198')
plt.ylabel('199')
plt.title('Performance Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

#X = (df - int(df.mean()))/(int(df.max()) - int(df.min()))
"""Normalizing the Data"""
X = (df - df.mean())/(df.max() - df.min())
#print(X.describe())

y = df['199']
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

X_train = X_train.T                 #Transposed to make is 2D
y_train = np.array([y_train])       #Stored in as NumPy array
y_test = np.array([y_test])         #Stored in as NumPy array

X_test = X_test.T                   #Transposed to make is 2D

#Linear modek called with the train, test, learning rate, and iterations as the parameters
linear_regression_model(X_train,y_train,X_test,y_test,0.1, 5000)
