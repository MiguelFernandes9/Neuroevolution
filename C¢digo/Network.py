import numpy as np
import math

class Network(object):

    def __init__(self, sizes):
        """
            "sizes" -> [1,2,3], where the 1st layer was 1 neuron and the other 2 have 2 and 3 respectively. Note that the 1st layer
            is the input layer.

            The Bias and the Weights are initialized at random with a Gaussian distribution with mean 0, and varience 1.

        """

        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

        # helper variables
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers-2)])

    def tanh(self,z):
        return np.tanh(z)

    def sigmoid(self, z):
        # The sigmoid function.
        return 1.0/(1.0+np.exp(-z))

    def output_nn(self, input_test):
        # Returns the output of the nn with a certain input
        output = input_test
        for b, w in zip(self.biases, self.weights):
            output = self.tanh(np.dot(w,output)+b)
        return output

    def softmax(self, output):
        
        sum_a = sum([math.pow(math.e,y) for y in output])

        output = [y/sum_a for y in output]

        return output


    def mean_error(self,X,y):
        total_score=0
        for i in range(X.shape[0]):
            predicted = self.output_nn(X[i].reshape(-1,1))
            actual = y[i]
            total_score += np.power(np.argmax(predicted)-actual,2)/2  # mean-squared error
        return total_score


    def accuracy(self, X, y):
        accuracy = 0
        for i in range(X.shape[0]):
            output = self.output_nn(X[i].reshape(-1,1))
            #print("Raw Output:", str(output), " Real Output: ", str(np.argmax(output)), " target: ", y[i], "\n")
            if(int(np.argmax(output)) == y[i]):
                #print(accuracy)
                accuracy += 1
        #print("accuracy:", accuracy)
        return accuracy / len(y) * 100

    def __str__(self):
        s = "\nBias:\n\n" + str(self.biases)
        s += "\nWeights:\n\n" + str(self.weights)
        s += "\n\n"
        return s