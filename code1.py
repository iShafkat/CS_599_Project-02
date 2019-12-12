# This code creates neural network with dropout method and 
#evaluates the classification error with respect to the dataset size for MNIST dataset. This is a Digit Recognition Classifiaction Problem


# The following codes import libraries
import tensorflow as tf
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras import losses
from keras.layers import LocallyConnected1D, LocallyConnected2D, Conv2D
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dropout

def main():
# This is the Main Function of the Code
# Declaring The variables
    error_table =[]
    size =1000
    m=1000
    size_table =[]
    # Load dataket to appropriate valiables
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    for size in range(100000):
        
        X = x_train[0:m,:]  # This commands stores the input features of each observation              
        y = y_train[0:m]
        y = to_categorical(y)                  # Categorizes the training labels  
        test_label1 = y_test        # This commands stores the input labels of each observation
        test_array = x_test      # This commands stores the input features of each observation
        test_label = to_categorical(test_label1)                                  # Categorizes the training labels
        classification_error =network_one(X,y,test_label1,test_array)                     # Calls network one
        error_table.append(classification_error)
        size =size*5
        size_table.append(size)
        m =m*5
        
    # The following code Plots The Figure

    plt.ylabel('Classification Error %')
    plt.xlabel('Dataset Size')
    plt.plot(size_table, error_table) # This line plots the output figure
    plt.show()                                   # This line prints out the output figure
    plt.savefig('Output.png')      # This line saves the output figure
        
        
        
        
def network_one(X,y,test_label1,test_array):
    accuracy_one, epoch =[],30
    model = Sequential()
    X= X.reshape(X.shape[0], 28, 28,1).astype('float32')
    model.add(Conv2D(10,(3,3), input_shape=(28,28,1), activation='sigmoid'))       
    model.add(LocallyConnected2D(2084, (5,5), activation='sigmoid'))
    model.add(Flatten())
    model.add(Dropout(0.5)) 
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss=losses.mean_squared_error, optimizer='sgd', metrics=['accuracy'])   # Compiles the network
    model.fit(X, y, epochs=epoch, batch_size=10)         # Trains the network
    _, accuracy = model.evaluate(X, y)                   # Training Accuracy is measured
    predictions = model.predict_classes(test_array)      # The network Predicts the output of the test data set
    correct =0
   # Finds the accuracy of the network for test data
    for i in range(len(predictions)):
        if predictions[i]==test_label1[i]:
            correct +=1
        else:
            pass
    accuracy =100 * correct/len(predictions)
    classification_error = 100-accuracy
    return classification_error

# This calls the main function
if __name__ == '__main__':
    main()
