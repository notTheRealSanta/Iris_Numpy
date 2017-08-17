
import numpy as np
import pandas as pd

np.random.seed(1)

#sigmoid function
def sigmoid ( x ) :
    ret = 1/(1+np.exp(-x))
    return ret

def der_sigmoid ( x ) :
    ret = x * ( 1 - x )
    return ret

#reading iris dataset with pandas
dataset = pd.read_csv('iris.csv', header = None)

values = dataset.values[:,0:4].astype(float)
labels_name = dataset.values[:,4:5]
labels = np.zeros ( (150,3) )

#one-hot encoding
for x in range(0,150):
	if ( labels_name[x]  == 'Iris-setosa') :
		labels[x,0] = 1
	elif ( labels_name[x]  == 'Iris-versicolor') :
		labels[x,1] = 1
	else :
		labels[x,2] = 1

#print ( "Input shape : ",values.shape ,"\nOutput shape :",labels.shape)

synapse_1 = 2 * np.random.random((4,12)) - 1
synapse_2 = 2 * np.random.random((12,3)) - 1

#print ( "Synapse 1 shape : ",synapse_1.shape)
#print ( "Synapse 2 shape : ",synapse_2.shape)
#make the values and labels as 2d array
for l in range(1000) :

    for i in range(150) :

        layer_0 = values[i]
        layer_0 = np.reshape(layer_0,(1,4))
        layer_1 = sigmoid (np.dot(layer_0,synapse_1))

        layer_2 = sigmoid(np.dot(layer_1, synapse_2))

        layer_2_error = layer_2 - labels[i]

        layer_2_delta = layer_2_error * der_sigmoid(layer_2)

        synapse_2_der = np.dot(layer_1.T,layer_2_delta)

        synapse_2 -= synapse_2_der

layer_0 = values[101]
layer_0 = np.reshape(layer_0,(1,4))
layer_1 = sigmoid (np.dot(layer_0,synapse_1))

layer_2 = sigmoid(np.dot(layer_1, synapse_2))

print (layer_2)
