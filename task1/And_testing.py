# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:35:47 2020

@author: bhaumik
"""

import numpy as np
from perceptron import Perceptron


training_inputs=[]
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))

labels=np.array([1,0,0,0])

perceptron = Perceptron(2)

perceptron.train(training_inputs,labels)

inputs=np.array([1,1])
print(perceptron.predict(inputs))

inputs=np.array([0,0])
print(perceptron.predict(inputs))

# Conclusion :
    
# Here i have taken learning rate 100 so the change in weights will be in the multiple of 100
# and threshold decidea how many times it ierates

# output
# bias 0.0
# weights [0. 0.]
# bias 100.0
# weights [100. 100.]
# bias 0.0
# weights [  0. 100.]
# bias -100.0
# weights [0. 0.]
# bias -100.0
# weights [0. 0.]
# bias 0.0
# weights [100. 100.]
# bias -100.0
# weights [  0. 100.]
# bias -100.0
# weights [  0. 100.]
# bias -100.0
# weights [  0. 100.]
# bias 0.0
# weights [100. 200.]
# bias -100.0
# weights [  0. 200.]
# bias -200.0
# weights [  0. 100.]
# bias -200.0
# weights [  0. 100.]
# bias -100.0
# weights [100. 200.]
# bias -100.0
# weights [100. 200.]
# bias -200.0
# weights [100. 100.]
# bias -200.0
# weights [100. 100.]
# bias -100.0
# weights [200. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# bias -200.0
# weights [100. 200.]
# 1
# 0