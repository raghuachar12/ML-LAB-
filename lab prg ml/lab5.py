import numpy as np

x = np.array(([2,9],[1,5],[3,6]), dtype = float)

y = np.array(([92],[86],[89]),dtype = float)

x = x/np.amax(x,axis = 0)
y = y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

epoch = 7000
inputlayer_neurons = 2
hiddenlayer_neurons = 3
output_neurons = 1

wh = np.random.uniform(size = (inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size = (1,hiddenlayer_neurons))
wout = np.random.uniform(size = (hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size = (1, output_neurons))

for i in range(epoch):
    hinp1 = np.dot(x,wh)
    hinp = hinp1+bh
    hlayer_act = sigmoid(hinp)
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1+bout
    output = sigmoid(outinp)

print('\nInput: \n' + str(x))
print('\nActual Output: \n' + str(y))
print('\nPredicted Output: \n', output)