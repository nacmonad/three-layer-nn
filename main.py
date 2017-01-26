#!/usr/bin/python

# forked from https://github.com/miloharper/multi-layer-neural-network/blob/master/main.py
# submitted for the Intro to Deep Learning #2 Challenge on Siraj Raval's Youtube channel
# https://www.youtube.com/watch?v=p69khggr1Jo

from numpy import exp, array, random, dot


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2, layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2, output_from_layer_3 = self.think(training_set_inputs)
           

            # Back propogate to adjust them weights!
            # Calculate the error for layer 3 (The difference between the desired output
            # and the predicted output)
            layer3_error = training_set_outputs - output_from_layer_3
            layer3_delta = layer3_error * self.__sigmoid_derivative(output_from_layer_3)
            #print(layer3_error)
            
            # Calculate the error for layer 2 (By looking at the weights in layer 2,
            # we can determine by how much layer 2 contributed to the error in layer 3).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment
            self.layer3.synaptic_weights += layer3_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        output_from_layer3 = self.__sigmoid(dot(output_from_layer2, self.layer3.synaptic_weights))
        return output_from_layer1, output_from_layer2, output_from_layer3

    # The neural network prints its weights
    def print_weights(self):
        print "    Layer 1 (3 neurons, each with 3 inputs): "
        print self.layer1.synaptic_weights
        print "    Layer 2 (3 neuron, with 3 inputs):"
        print self.layer2.synaptic_weights
        print "    Layer 3 (1 neuron, with 3 inputs):"
        print self.layer3.synaptic_weights

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # THE NNET
    # Create layer 1 (3 neurons, each with 3 inputs)
    layer1 = NeuronLayer(3, 3)
    # Create layer 2 (3 single neuron with 3 inputs)
    layer2 = NeuronLayer(3, 3)
    # Output layer 3 (1 single neuron with 3 inputs)
    layer3 = NeuronLayer(1, 3)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2, layer3)

    print "Stage 1) Random starting synaptic weights: "
    neural_network.print_weights()

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 20000)

    print "Stage 2) New synaptic weights after training: "
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print "Stage 3) Considering a new situation [1, 1, 0] -> ?: "

    output = neural_network.think(array([1, 1, 0]))
    # output has outputs to all layers, show the output layer (3rd layer)
    print output[2]
    
