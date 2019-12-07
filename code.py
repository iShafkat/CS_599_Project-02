# This code creates neural network with dropout method and 
#evaluates the classification effor with respect to the dataset size for MNIST dataset.

#The following code imports libraries 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
import tensorflow as tf
import matplotlib.pyplot as plt

def create_model():
    # This function creates the neural network model 
        
    model = Sequential()
    #model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(784, activation='relu', kernel_constraint=maxnorm(3)))  # This is the input Layer
    model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3))) # This is the first hidden layer
    model.add(Dropout(0.5))                                                 # Dropout percentage for the first hidden layer
    model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3))) # This is the Second hidden layer
    model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3))) # This is the Third hidden layer
    model.add(Dense(10, activation='sigmoid'))                              # This is the Output Layer
    # Compile model
    sgd = SGD(lr=0.1, momentum=0.9)                                         # This line of code optimizes the model
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) #This evaluates the training accuracy
    return model


def main():
    epoch =10
    correct =0
    size = 100
    classification_error, dataset_size =[], []
    for size in range(10000):
        # load dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Create Model
        x_train = x_train[:,1:size]
        model =create_model()
        # Fit Training Data to the model
        model.fit(x_train, y_train, epochs=epoch, batch_size=10)
        # Calculate The Training Accuracy
        _, accuracy = model.evaluate(x_train, y_train)
        # Predict the test data
        predictions = model.predict_classes(x_test)
    
# The following code is used to calculate the test error

	correct =0
        error =[]
        for i in range(len(predictions)):
            if predictions[i]==y_test[i]:
                correct +=1
            else:
                pass
            accuracy =100 * correct/len(predictions)
            error =100-accuracy
            epoch +=1
        classification_error.append(error)
        dataset_size.append(size)
    
# The following code Plots The Figure

    plt.ylabel('Classification Error %')
    plt.xlabel('Dataset Size')
    plt.plot(classification_error, dataset_size) # This line plots the output figure
    plt.show()                                   # This line prints out the output figure
    plt.savefig('output.png')			 # This line saves the output figure

if __name__ == '__main__':
    main()
