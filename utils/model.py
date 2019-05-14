"""
Neural Network model script
It takes care of training the model for given specification & data

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
from utils.layers import *
from utils.loss import reg_loss_calc

class NeuralNetwork:
    def __init__(self,X,y,validation_X,validation_y,D,K,h_list,p_list,
                 activation_list,output_activation,scaling,const=0.1):
        # parameters
        self.X = X
        self.y = y
        self.validation_X = validation_X
        self.validation_y = validation_y
        self.nb_layers = len(h_list) + 1
        nb_layers = self.nb_layers
        self.h_list = h_list
        self.p_list = p_list
        self.W_list = [None] * nb_layers
        self.b_list = [None] * nb_layers
        self.X_list = [None] * nb_layers
        self.X_list[0] = self.X

        self.act_list = [None] * nb_layers
        self.lin_list = [None] * nb_layers
        self.drop_list = [None] * nb_layers
        self.linout_list = [None] * nb_layers
        self.actout_list = [None] * nb_layers
        self.mask_list = [None] * nb_layers

        self.dscores_list = [None] * nb_layers
        self.dW_list = [None] * nb_layers
        self.db_list = [None] * nb_layers
    
        # randomize weights
        for i in range(nb_layers):
            # scaling term
            if scaling=='constant':
                scaler = const
            elif scaling=='xavier':
                scaler = np.sqrt(1.0 / h_list[i - 1])
            elif scaling=='he':
                scaler = np.sqrt(2.0 / h_list[i - 1])
            else:
                raise ValueError('Can only use "constant","xavier" or "he" scaling')
            # linear layers
            self.lin_list[i] = Linear()
            if (i == 0):
                self.W_list[i] = np.random.randn(D,h_list[i])
                self.b_list[i] = np.zeros((1,h_list[i]))
            elif (i==nb_layers-1):
                self.W_list[i] = np.random.randn(h_list[i-1],K) * scaler
                self.b_list[i] = np.zeros((1,K))
            else:
                self.W_list[i] = np.random.randn(h_list[i-1],h_list[i]) * scaler
                self.b_list[i] = np.zeros((1,h_list[i]))

            # activation layers
            seed = random.randint(1, 9999)
            if (i<nb_layers-1):
                self.act_list[i] = activation_list[i]
                self.drop_list[i] = Dropout(seed=seed,p=p_list[i])
            else:
                self.act_list[i] = output_activation
                self.drop_list[i] = Dropout(seed=seed,p=1)
    
    def summary(self):
        nb_layers = self.nb_layers
        h_list = self.h_list
        N = self.X_list[0].shape[0]
        
        for i in range(nb_layers):
                
            print('[LIN] {}: Input ({},{}), Weight {}, Bias {}, Output ({},{})'
                .format(i,N,self.W_list[i].shape[0],                                                              
                self.W_list[i].shape,
                self.b_list[i].shape,
                N,self.W_list[i].shape[1]))
            
            if (i < nb_layers - 1):
                print('[ACT] {}: Input-Output ({},{}), Activation {}, Dropout {},'
                    .format(i,N,self.W_list[i].shape[1],
                        self.act_list[i],self.p_list[i]))
            else:
                print('[OUT] {}: Input-Output ({},{}), Activation {},'
                    .format(i,N,self.W_list[i].shape[1],self.act_list[i]))
            
    def train(self,iter,reg,step_size,use_l2):
        nb_layers = self.nb_layers
        X_list = self.X_list
        act_list = self.act_list
        lin_list = self.lin_list
        actout_list = self.actout_list
        linout_list = self.linout_list
        dscores_list = self.dscores_list
        dW_list = self.dW_list
        db_list = self.db_list
        y = self.y
        
        for j in np.arange(iter):
            # forward
            for i in range(nb_layers):
                linout_list[i] = lin_list[i].forward(X_list[i],
                    self.W_list[i],self.b_list[i])
                actout_list[i] = act_list[i].forward(linout_list[i])
                # dropout
                self.mask_list[i] = self.drop_list[i].mask(actout_list[i])
                actout_list[i] *= self.mask_list[i]
                if (i < nb_layers-1): X_list[i+1] = actout_list[i]

            if type(act_list[nb_layers-1]) is Softmax:
                data_loss = act_list[nb_layers-1].cross_entropy(actout_list[nb_layers-1],y)
            reg_loss = reg_loss_calc(self.W_list,l2=use_l2,reg=reg)
            loss = data_loss + reg_loss

            if j % (iter/10) == 0:
                print('Training Loss at iteration {}: {}'.format(j,loss))

            
            # backward
            for i in reversed(range(nb_layers)):
                # backward activation
                if i < nb_layers-1:
                    dscores_list[i] = act_list[i].backward(X_list[i+1],
                        dscores_list[i+1].dot(self.W_list[i+1].T))
                    # dropout
                    dscores_list[i] *= self.mask_list[i]
                else:
                    dscores_list[i] = act_list[i].backward(linout_list[i],y)   

                # backward linear
                dW_list[i],db_list[i] = lin_list[i].backward(X_list[i],
                    self.W_list[i],dscores_list[i],reg=reg)

            if j % (iter/10) == 0:
	            for i in range(nb_layers):
	            	if i==0:
	            		linout_list[i] = lin_list[i].forward(self.validation_X,
                            self.W_list[i],self.b_list[i])
	            	else:
	            		linout_list[i] = lin_list[i].forward(X_list[i],
                            self.W_list[i],self.b_list[i])
	            	actout_list[i] = act_list[i].forward(linout_list[i])
	            	if (i < nb_layers-1): X_list[i+1] = actout_list[i]
	            if type(act_list[nb_layers-1]) is Softmax:
	                data_loss = act_list[nb_layers-1].cross_entropy(
                        actout_list[nb_layers-1],self.validation_y)
	            else:
	                data_loss = act_list[nb_layers-1].hinge(actout_list[nb_layers-1],
                        self.validation_y)
	            reg_loss = reg_loss_calc(self.W_list,l2=use_l2,reg=reg)
	            loss = data_loss + reg_loss
	            print('Validation Loss at iteration {}: {}'.format(j,loss))

            # update
            step_size=step_size*0.9999
            for i in range(nb_layers):
                self.W_list[i] -= step_size * dW_list[i]
                self.b_list[i] -= step_size * db_list[i]  

    def pred(self,X):
        nb_layers = self.nb_layers
        X_list = self.X_list
        X_list[0] = X
        act_list = self.act_list
        lin_list = self.lin_list
        actout_list = self.actout_list
        linout_list = self.linout_list
        W_list = self.W_list
        b_list= self.b_list
        
        for i in range(nb_layers):
            linout_list[i] = lin_list[i].forward(X_list[i],W_list[i],b_list[i])
            actout_list[i] = act_list[i].forward(linout_list[i])
            if (i < nb_layers-1): X_list[i+1] = actout_list[i]
        pred = np.argmax(actout_list[nb_layers-1], axis=1)
        return pred