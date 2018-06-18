#Basic Neural Network
#Layers{
#       1-input(l0),
#       2-hidden(l1,l2),
#       1-output(l3)
#   }


import numpy as np

#create function(sigmoid)
def sigm(x, derive=False):
    if(derive == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


#input data
inp_data = np.array([[0,0,1],
                     [1,1,1],
                     [1,0,1],
                     [0,1,1]])

#output data
out_data = np.array([[0,1,1,0]]).T

np.random.seed(1)

#generating weights
weig1 = 2*np.random.random((3,4)) - 1
weig2 = 2*np.random.random((4,3)) - 1
weig3 = 2*np.random.random((3,1)) - 1


for _ in range(100000):

    #front feeding
    l0 = inp_data
    l1 = sigm(l0.dot(weig1))
    l2 = sigm(l1.dot(weig2))
    l3 = sigm(l2.dot(weig3))

    #training
    #error = sum(output.dot(weights))*derivative(output)
    #output === output_layer
    l3_error = out_data - l3
    l3_delta = l3_error*sigm(l3,True)
    l2_error = l3_delta.dot(weig3.T)
    l2_delta = l2_error*sigm(l2, True)
    l1_error = l2_delta.dot(weig2.T)
    l1_delta = l1_error*sigm(l1,True)

    #updating weghts
    #weigh += dot(simultaneousLayers)
    weig3 += l2.T.dot(l3_delta)
    weig2 += l1.T.dot(l2_delta)
    weig1 += l0.T.dot(l1_delta)
    


print("Output")
print(l0)
print(l1)
print(l2)
print(l3)

