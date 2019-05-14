"""
main script
It loads the dataset, model and ceates an instance of the model.

MIT License

Copyright (c) 2018 Vaibhav Bhilare

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# imports
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets

from utils.model import NeuralNetwork
from utils.layers import *
import utils.loss

def main():
    # Load iris dataset from scikit-learn.datasets
    iris = datasets.load_iris()
    # Separate samples and labels
    X = iris.data
    y=iris.target
    X=np.array(X)
    ## Preprocessing
    # Normalize the dataset
    mean_x=np.mean(X,axis=0)
    sd_x=np.sqrt(np.var(X,axis=0))
    X=np.divide((X-mean_x),sd_x)
    y=np.array(y)
    X_Length,_=X.shape
    # Shufle the dataset
    ranges=random.sample(range(0, X_Length), X_Length)
    X=X[ranges]
    y=y[ranges]
    # Train-Validation-Test split
    train_X = X[:int(len(X)*0.8)]
    validation_X = X[int(len(X)*0.8):int(len(X)*0.9)]
    test_X = X[int(len(X)*0.9):]
    train_y = y[:int(len(y)*0.8)]
    validation_y = y[int(len(y)*0.8):int(len(y)*0.9)]
    test_y = y[int(len(y)*0.9):]
    # N = Number of samples, D=Features
    N,D = X.shape
    # K= Classification Classes
    K = 3 #3 classes
    print('Shape of X:',X.shape)
    print('Shape of y:',y.shape)

    # Create the model
    model = NeuralNetwork(train_X,train_y,validation_X,validation_y,D,K,
        h_list=[20,20],p_list=[1,0.5],activation_list=[Sigmoid(),Sigmoid()],
        output_activation=Softmax(),scaling='xavier')
    # Print model summary
    model.summary()
    # Train the model
    model.train(iter=25000,reg=0,step_size=1e-2,use_l2=True)
    # Test the model
    test_pred=model.pred(test_X)
    correct=np.sum(test_pred==test_y)
    correct_per=correct/len(test_y)
    # Print Test accuracy
    print('Accuracy: {}%'.format(correct_per*100))

if __name__ == '__main__':
    main()