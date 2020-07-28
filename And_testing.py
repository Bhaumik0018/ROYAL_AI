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
    
#     With different Learning rate the addtion or difference magnitude is altered
#     in the multiples of learning rate here it is 10 and with different threshold the iterations are controlled here 10 iterations
#     are printed.
#output
# 0
# bias 0.0
# weights [0. 0.]
# bias 10.0
# weights [10. 10.]
# bias 0.0
# weights [ 0. 10.]
# bias -10.0
# weights [0. 0.]
# 1
# bias -10.0
# weights [0. 0.]
# bias 0.0
# weights [10. 10.]
# bias -10.0
# weights [ 0. 10.]
# bias -10.0
# weights [ 0. 10.]
# 2
# bias -10.0
# weights [ 0. 10.]
# bias 0.0
# weights [10. 20.]
# bias -10.0
# weights [ 0. 20.]
# bias -20.0
# weights [ 0. 10.]
# 3
# bias -20.0
# weights [ 0. 10.]
# bias -10.0
# weights [10. 20.]
# bias -10.0
# weights [10. 20.]
# bias -20.0
# weights [10. 10.]
# 4
# bias -20.0
# weights [10. 10.]
# bias -10.0
# weights [20. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# 5
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# 6
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# 7
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# 8
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# 9
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# bias -20.0
# weights [10. 20.]
# 1
# 0


