import numpy as np

class Artificial_Neural_Network(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # initialize weights
        # numpy.random.normal(loc=0.0, scale=1.0, size=None)
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5, (self.output_nodes, self.hidden_nodes))
        self.learning_rate = learning_rate
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def ann_forward_pass(self, inputs, targets):
        # convert list to array
        inputs = np.array(inputs).T  # transpose
        # Forward pass
        hiddens_in = np.dot(self.weights_input_to_hidden, inputs)
        hiddens_out = self.activation_function(hiddens_in)
        outputs = np.dot(self.weights_hidden_to_output, hiddens_out)
        loss=np.mean(np.sqrt(targets-outputs))
        return loss

    def ann_backward_propagation(self, inputs, hiddens_out, outputs, targets):
        # convert list to array
        targets = np.array(targets).T
        # backward propagation
        outputs_err = targets - outputs
        hiddens_err = np.dot(self.weights_hidden_to_output.T, outputs_err)
        hiddens_grad = hiddens_out * (1 - hiddens_out)
        self.weights_hidden_to_output += self.learning_rate * outputs_err * hiddens_out.T
        self.weights_input_to_hidden += self.learning_rate * np.dot((hiddens_err * hiddens_grad), inputs.T)
        return self.weights_input_to_hidden, self.weights_hidden_to_output



class Recurrent_Neural_Network(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, sequence_length, learning_rate):
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        self.sequence_length=sequence_length   # =len(inputs)
        self.learning_rate=learning_rate
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5, (self.output_nodes, self.hidden_nodes))
        self.bias_hidden=np.zeros((hidden_nodes, 1))
        self.bias_output=np.zeros((output_nodes, 1))
        self.momentum_weights_input_to_hidden=np.zeros_like(self.weights_input_to_hidden)
        self.momentum_weights_hidden_to_hidden=np.zeros_like(self.weights_hidden_to_hidden)
        self.momentum_weights_hidden_to_output=np.zeros_like(self.weights_hidden_to_output)
        self.momentum_bias_hidden=np.zeros_like(self.bias_hidden)
        self.momentum_bias_output=np.zeros_like(self.bias_output)
        self.input_states={}
        self.hidden_states={}
        self.output_states={}
        self.loss=0

    def rnn_forward_pass(self, inputs, initial_hidden_state, targets):
        self.hidden_states[-1]=np.copy(initial_hidden_state)     # hidden states are in reversed order
        for i in range(len(inputs)):
            self.input_states[i]=np.zeros((self.input_nodes, 1))
            self.input_states[i]=inputs[i]
            self.hidden_states[i]=np.tanh(np.dot(self.weights_input_to_hidden, self.input_states[i])+
                                            np.dot(self.weights_hidden_to_hidden, self.hidden_states[i-1])+self.bias_hidden)
            self.output_states[i]=np.dot(self.weights_hidden_to_output, self.hidden_states[i])+self.bias_output
            self.loss+=np.square(targets[i]-self.output_states[i])/2.
        return self.loss

    def rnn_backward_propagation(self, inputs, targets):
        derivative_weights_input_to_hidden=np.zeros_like(self.weights_input_to_hidden)
        derivative_weights_hidden_to_hidden=np.zeros_like(self.weights_hidden_to_hidden)
        derivative_weights_hidden_to_output=np.zeros_like(self.weights_hidden_to_output)
        derivative_bias_hidden=np.zeros_like(self.bias_hidden)
        derivative_bias_output=np.zeros_like(self.bias_output)
        derivative_hidden_next=np.zeros_like(self.hidden_states[0])
        for i in reversed(range(len(inputs))):
            derivative_output=targets[i]-self.output_states[i]
            derivative_weights_hidden_to_output+=np.dot(derivative_output, self.hidden_states[i].T)
            derivative_bias_output+=derivative_output
            derivative_hidden=np.dot(self.weights_hidden_to_output.T, derivative_output)+derivative_hidden_next
            derivative_tanh=(1-self.hidden_states[i]*self.hidden_states[i])*derivative_hidden
            derivative_bias_hidden+=derivative_tanh
            derivative_weights_input_to_hidden+=np.dot(derivative_tanh, self.input_states[i].T)
            derivative_weights_hidden_to_hidden+=np.dot(derivative_tanh, self.hidden_states[i-1].T)
            derivative_hidden_next=np.dot(self.weights_hidden_to_hidden.T, derivative_tanh)
        for params in [derivative_weights_input_to_hidden, derivative_weights_hidden_to_hidden, derivative_weights_hidden_to_output,
                       derivative_bias_hidden, derivative_bias_output]:
            np.clip(params, -5, 5, out=params)    # clip the derivatives for mitigating explosion
        for param, derivative_param, momentum_param in zip([self.weights_input_to_hidden, self.weights_hidden_to_hidden,
                                                            self.weights_hidden_to_output, self.bias_hidden, self.bias_output],
                                                           [derivative_weights_input_to_hidden, derivative_weights_hidden_to_hidden,
                                                            derivative_weights_hidden_to_output, derivative_bias_hidden, derivative_bias_output],
                                                           [self.momentum_weights_input_to_hidden, self.momentum_weights_hidden_to_hidden,
                                                            self.momentum_weights_hidden_to_output, self.momentum_bias_hidden, self.momentum_bias_output]):
            momentum_param+=derivative_param*derivative_param
            param+=-self.learning_rate*derivative_param/np.sqrt(momentum_param+1e-8)
        return self.weights_input_to_hidden, self.weights_hidden_to_hidden, self.weights_hidden_to_output, \
               self.bias_hidden, self.bias_output, self.hidden_states[len(inputs)-1]


#print(np.tanh(0.49999999999999994))
#print(np.arctanh(0.46211715726000974))