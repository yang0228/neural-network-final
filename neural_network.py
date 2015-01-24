import math
import numpy as np
import numpy.linalg.linalg as la

class NeuralNetwork:

    """
    A Neural Network implementation with back propagation
    """
    
    def __init__(self,n_inputs,n_outputs,n_neurons,n_hidden_layers,n_steps,step_size,step_decay,weight_bound,bias_bound):
        """
        Constructor - sets internal properties and initialises weights and biases
        :param n_inputs:
        :param n_outputs:
        :param n_neurons:
        :param n_hidden_layers:
        :return:
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.n_layers = n_hidden_layers + 2
        self.n_steps = n_steps
        self.step_size = step_size
        self.step_decay = step_decay

        # sets the weight/bias symmetry
        self.weight_bound = weight_bound
        self.bias_bound = weight_bound

        self.weights = []
        self.biases = []
        self.activations = []
        self.activations_d = []
        
        self._initialise_weights_and_bias()
        
    def _initialise_weights_and_bias(self):
        """
        Initialises weights and biases
        Note, each layer contains the w and b to get to the NEXT layer
        :return:
        """
        
        # input to hidden
        # self.weights.append(np.random.rand(self.n_inputs,self.n_neurons))
        # self.biases.append(np.random.rand(self.n_neurons,1))

        self.weights.append(np.random.uniform(-self.weight_bound,self.weight_bound,(self.n_inputs,self.n_neurons)))
        self.biases.append(np.random.uniform(-self.bias_bound,self.bias_bound,(self.n_neurons,1)))
        
        # hidden to hidden
        for x in range(0,self.n_hidden_layers-1):
            self.weights.append(np.random.uniform(-self.weight_bound,self.weight_bound,(self.n_neurons,self.n_neurons)))
            self.biases.append(np.random.uniform(-self.bias_bound,self.bias_bound,(self.n_neurons,1)))
        
        # hidden to output
        self.weights.append(np.random.uniform(-self.weight_bound,self.weight_bound,(self.n_neurons,self.n_outputs)))
        self.biases.append(np.random.uniform(-self.bias_bound,self.bias_bound,(self.n_outputs,1)))
        
    def activation(self,x):
        """
        The activation function
        :param x:
        :return:
        """
        return math.tanh(x)
    
    def activation_d(self,x):
        """
        The activation derivative function
        :param x:
        :return:
        """
        return 1.0 - math.tanh(x)**2
    
    def reset(self):
        """
        Resets activations etc for the next iteration
        :return:
        """
        self.activations = []
        self.activations_d = []
        
    def prepare_instance(self,instance,n_labels):
        """
        Prepares an instance for use with the model
        :param instance:
        :param n_labels:
        :return:
        """
        label = int(instance[0])
        inputs = map(float,instance[1:])
        outputs = [-1.0]*n_labels
        outputs[label] = 1.0
        return self.vectorise(np.array(inputs)),self.vectorise(np.array(outputs)),label

    def vectorise(self,vector):
        """
        Ensures that the vector is a column vector
        :param vector:
        :return:
        """
        length = max(vector.shape)
        vector.shape = (length ,1)
        return vector
    
    def feed_forward(self,inputs):
        """
        Forward-feeds the network
        :param inputs:
        :return:
        """
        self.activations.append(inputs)
        self.activations_d.append(np.array([]))

        for i in range(0,self.n_hidden_layers + 1):

            weights = self.weights[i]
            biases = self.vectorise(self.biases[i])

            signal = self.activations[-1]
            signal = la.dot(signal.T,weights)
            signal = la.add(signal.T,biases)

            # dropout regularisation - ruined performance
            dropout_vector = self.dropout_vector(len(signal),0.5)

            activation = self.vectorise(np.array([self.activation(x[0]) for x in signal]))
            # print(activation)
            # activation = la.multiply(activation,dropout_vector)
            # print(activation)
            # exit()
            self.activations.append(activation)

            activation_d = self.vectorise(np.array([self.activation_d(x[0]) for x in activation]))
            # activation_d = la.multiply(activation_d,dropout_vector)
            self.activations_d.append(activation_d)
    
    def back_propagate(self,actual,step):
        """
        Back-propagates the network, gradient descent
        :param actual:
        :param step:
        :return:
        """
        delta = self.activations[-1] - actual

        # for i in range(self.n_layers-2,1,-1):
        for i in range(self.n_layers-1,0,-1):

            delta = self.vectorise(delta)
            prev_activations = self.vectorise(self.activations[i-1])
            prev_activations_d = self.vectorise(self.activations_d[i-1])

            # print("BP %d -> %d" % (i,i-1))

            # update the bias (v)
            self.biases[i-1] = np.subtract(self.biases[i-1],la.multiply(step,delta))

            # update the weights (previous activation^T x delta
            weight_delta = la.multiply(prev_activations.T,delta) # matrix
            self.weights[i-1] = np.subtract(self.weights[i-1],la.multiply(step,weight_delta).T)

            # calculate new delta
            # delta (v) x weights of current layer (m)_T (dot)
            weights = self.weights[i-1]
            signal = la.dot(delta.T,weights.T).T

            if i > 1:
                delta = la.multiply(prev_activations_d,signal)


    def train(self,data):
        """
        Trains the network
        :param data:
        :param n_steps:
        :param step_size:
        :param step_decay:
        :param n_labels:
        :return:
        """
        step_size = self.step_size
        for i in range(0,self.n_steps):
            
            for x in range(0,len(data)):
                inputs,outputs,label = self.prepare_instance(data[x],self.n_outputs)
                self.feed_forward(inputs)
                self.back_propagate(outputs,self.step_size)
                self.reset()
            
            step_size *= self.step_decay
            
    def predict(self,inputs):
        """
        Returns the predicted label
        :param inputs:
        :return:
        """
        self.reset()
        self.feed_forward(inputs)
        activations = self.activations[-1]
        
        label = 0
        max_score = -1.0
        
        for i in range(0,len(activations)):
            if activations[i] > max_score:
                label = i
                max_score = activations[i]

        self.reset()

        return label, max_score, activations

    def predict_all(self,data):
        """
        Runs predictions on a full swath of data
        :param data:
        :return:
        """
        predictions = []
        for x in range(0,len(data)):
            inputs, outputs, label = self.prepare_instance(data[x],self.n_outputs)
            prediction, max_score, activations = self.predict(inputs)
            predictions.append((prediction,label))

        return predictions

    def predict_fitness(self,data):

        """
        Runs predictions, establishing fitness
        :param data:
        :return:
        """

        total = 0.0
        correct = 0.0

        for x in range(0,len(data)):
            inputs, outputs, label = self.prepare_instance(data[x],self.n_outputs)
            prediction, max_score, activations = self.predict(inputs)
            total += 1.0
            if prediction - label == 0:
                correct += 1.0

        return correct/total

    def dropout_vector(self,length,p):
        """
        Produces a dropout vector for dropout regularisation
        :param length:
        :param p:
        :return:
        """
        dropouts = []
        for i in range(0,length):
            d = np.random.random()
            if d <= p:
                dropouts.append(0.0)
            else:
                dropouts.append(1.0 / 1.0 - p)

        return self.vectorise(np.array(dropouts))
