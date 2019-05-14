"""
layers script
This script contains definitions for all layers in the model

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
## Linear Layer
class Linear:
    def forward(self,X,W,b):
        #X: N x D
        #W: D x K
        scores = X.dot(W) + b #N x K
        return(scores)
    def backward(self,X,W,previous,reg=1e-6):
        #previous: N x K
        dW = X.T.dot(previous)
        dW += reg * W
        db = np.sum(previous,axis=0,keepdims=True)
        return dW,db

## Sigmoid Activation
class Sigmoid:

    def forward(self,scores):
        # scores: N x K
        out = 1.0 / (1.0 + np.exp(-1*(scores)))
        return out
    
    def backward(self,scores,previous):
        out = self.forward(scores)
        dscores = (1.0 - out) * out * previous
        return(dscores)

## ReLU
class ReLu:
    def forward(self,scores):
        # scores: N x K
        out = scores * (scores > 0)
        return out
    def backward(self, scores, previous):
        dscores = 1.0 * (scores > 0) * previous
        return(dscores)

## Dropout
class Dropout:
    def __init__(self,p,seed):
        self.seed = seed
        self.p = p
    def mask(self,scores,scaling=True):
        # if no dropout
        if self.p == 1: return np.ones_like(scores)
        
        np.random.seed(seed=self.seed)
        if (scaling): 
            scaler = 1.0/(1-self.p)
        else:
            scaler = 1.0
        mask = scaler * np.random.binomial(n=1,p=self.p,size=scores.shape)
        return(mask)

## Softmax
class Softmax:
    def forward(self,scores):
        exp_scores = np.exp(scores-np.max(scores))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return(probs)
    
    def backward(self,scores,y):
        N = scores.shape[0]
        dscores = self.forward(scores)
        dscores[range(N), y] -= 1
        dscores /= N
        return(dscores)   
    
    def cross_entropy(self,probs,y):
        # probs: N x K
        N = probs.shape[0]
        correct_logprobs = -np.log(probs[range(N),y])
        # data loss
        data_loss = np.sum(correct_logprobs) / N
        return(data_loss)