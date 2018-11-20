import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    return np.dot(1/np.sum(np.exp(z), axis=0), np.exp(z))
    
def forwardProp(z):
    #Layer 2 pre-activation
    z21 = W1[0,0]*z[0] + W1[0,1]*z[1] + b1[0]
    z22 = W1[1,0]*z[0] + W1[1,1]*z[1] + b1[1]

    z2 = np.array([[z21],[z22]])
    
    #Layer 2 activation
    a21 = sigmoid(z21)
    a22 = sigmoid(z22)

    a2 = np.array([[a21],[a22]])
    
    #Layer 3 pre-activation
    z31 = W2[0,0]*a21 + W2[0,1]*a22 + b2[0]
    z32 = W2[1,0]*a21 + W2[1,1]*a22 + b2[1]
    z33 = W2[2,0]*a21 + W2[2,1]*a22 + b2[2]

    z3 = np.array([[z31],[z32],[z33]])
    
    #Layer 3 activation
    outputActivation = softmax([[z31],[z32],[z33]])

    return z2,a2,z3,outputActivation 


def outputError(z):
    return -(y - z)

def hiddenLayerError(W, z, delta):
    return np.dot(W.transpose(), delta) * sigmoidGradient(z)

def gradientWeights(a,delta):
    return np.dot(delta, a.transpose())

#def gradientBias(delta):
#    return delta


#Values for our network
X = np.array([[0.1],[0.8]])
y = np.array([[0],[0],[1]])
W1 = np.array([[0.2, 0.5],[0.3, 0.4]])    
b1 = np.array([[0.1],[0.6]])
W2 = np.array([[0.1, 0.3],[0.7, 0.4],[0.01, 0.02]])
b2 = np.array([[0.2],[0.1],[0.4]])


#=======Backpropagation-Implementation===========

#Forward-Propagation
z2,a2,z3,output = forwardProp(X)

#Delta error of output-layer
delta3 = outputError(output)

#Delta error of layer 2:
delta2 = hiddenLayerError(W2,z2,delta3)

#Gradients W
gradW2 = gradientWeights(a2, delta3)
gradW1 = gradientWeights(X, delta2)

print('___________')
print('Gradients:')
print('W2')
print(gradW2)
print('___________')
print('W1')
print(gradW1)

#Gradients of the biases == errors of previous layer!!
print('___________')
print('b2')
print(delta3)
print('___________')
print('b1')
print(delta2)








