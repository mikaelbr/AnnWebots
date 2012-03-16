from math import exp

class Activation(object):
    """
        Class used to select the activation function. 
    """

    SIGMOID_LOG = 0
    SIGMOID_TANH = 1
    STEP = 2
    LINEAR = 3
    POS_LINEAR = 4

class Layer(object):



    def __init__(self, nodes, activation_function = Activation.SIGMOID_LOG):
        """
            1. the nodes that reside in the layer,
            2. the activation function shared by each of those nodes,
            3. the links entering the layer (i.e. links for which the 
                layer supplies post-synaptic neurons),
            4. the links exiting the layer (i.e. links for which the 
                layer supplied pre-synaptic neurons), and
            5. a learning-mode parameter indicating whether arcs that 
                enter the layer are currently amenable to synaptic (weight) 
                modification. The individual links will have a similar parameter 
                such that both the post-synaptic layer and the link of an arc must 
                be in a learning mode in order for plasticity to occur.
            6. a quiescent mode flag indicating whether or not the layer will be 
                run to quiescence during activation- level updates, as discussed below.
            7. an active flag indicating whether or not the layer is currently 
                able to a) update neuron activation levels, and b) send those signals 
                to downstream neurons.
            8. the maximum number of settling rounds used for runs to quiescence.
        """

        if isinstance(nodes, int):
            self.nodes = [Node(self) for x in range(nodes)]
        else:
            for node in nodes:
                node.layer = self

        # String representation of one of sigmoid (log/tanh), step and (pos_)linear
        self._activation_function_in = activation_function


        # Standard values (filled afterwords)
        self.entering = []
        self.exiting = []
        self.learning_mode = True

        self.quiescent_mode = True
        self.max_settling = 0

        # When summing the weighted inputs to a node, only include inputs 
        # from layers that are currently active.
        self.active = True




    def update(self):

        if not(self.active):
            return

        # Check quiescent mode ?
        # TODO: Implement quiescent mode

        # Not quiescent mode
        for node in self.nodes:
            node.activate()

    def activation_function (self, inpt):
        """
            Wrapper for the activation functions based 
            on the Activation type passed as argument
            to the constructor. 

            Input:
            'inpt' -> The weighted inputs from a node

            Output:
            the calculated activation level
        """

        if self._activation_function_in == Activation.SIGMOID_LOG:
            return self.sigmoid_log(inpt)

        if self._activation_function_in == Activation.SIGMOID_TANH:
            return self.sigmoid_tanh(inpt)

        if self._activation_function_in == Activation.STEP:
            return self.step(inpt)

        if self._activation_function_in == Activation.LINEAR:
            return self.linear(inpt)

        if self._activation_function_in == Activation.POS_LINEAR:
            return self.pos_linear(inpt)

        raise ValueError('No valid activation function passed to the layer')



    def sigmoid_log(self, inpt):
        return 1.0/(1.0 + exp(-inpt))

    def sigmoid_tanh(self, inpt):
        e = exp(2*inpt)
        return (e-1)/(e+1)

    def step(self, inpt, T = 0.5):
        return int(not inpt < T)

    def linear(self, inpt):
        return inpt

    def pos_linear(self, inpt):
        if inpt < 0:
            return 0.0
        return inpt
