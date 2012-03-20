from math import exp
from node import Node


class Activation(object):

    @staticmethod
    def sigmoid_log(inpt):
        """
            A sigmoid logistic function, which outputs values in the range [0,1].
        """
        return 1.0/(1.0 + exp(-inpt))

    @staticmethod
    def sigmoid_tanh(inpt):
        """
            A sigmoid tanh function, which outputs values in the range [-1, 1].
        """
        e = exp(2*inpt)
        return (e-1)/(e+1)

    @staticmethod
    def step(inpt, T = 0.5):
        """
            A step function with a threshold, T, that outputs values in the range [0,1]
        """
        return int(not inpt < T)

    @staticmethod
    def linear(inpt):
        """
            outputs the sum of weighted inputs
        """
        return inpt

    @staticmethod
    def pos_linear(inpt):
        """
            outputs the sum of weighted inputs when that sum is positive; otherwise it outputs a 0.
        """
        if inpt < 0:
            return 0.0
        return inpt

class Layer(object):
    """
        The general layer class houses a collection of nodes along 
        with several properties that pertain to each of them. It 
        also includes information concerning the links for which 
        it contributes pre-synaptic or post-synaptic neurons.
    """


    def __init__(self, name, nodes, activation_function = None, io_type=None):
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
        self.name = name
        self.type = io_type

        # 1. If the number of nodes wished is passed as argument, create nodes
        if isinstance(nodes, int):
            self.nodes = [Node(self) for n in range(nodes)]
        else:
            # List of nodes given. Set all to have current layer
            for node in nodes:
                node.layer = self

        # 2. Referance of one of sigmoid (log/tanh), step and (pos_)linear
        self.activation_function = activation_function

        # 7. When summing the weighted inputs to a node, only include inputs 
        # from layers that are currently active.
        self.active = True

        # 3., 4., 5., 6., 8. Standard values (filled afterwords)
        self.entering = []
        self.exiting = []
        self.learning_mode = False

        self.quiescent_mode = False
        self.max_settling = 0

    def __str__(self):
        return str(self.name)

    def update(self, quiescent_mode=None):

        if quiescent_mode == None:
            quiescent_mode = self.quiescent_mode

        if not(self.active):
            return

        if not quiescent_mode or self.max_settling < 1:
            # Not quiescent mode, update activation levels
            for node in self.nodes:
                node.activate()

            return

        # Is quiescent mode
        prev = []
        # Avoid infinite loop.
        for i in range(self.max_settling):            
            self.update(False) # Update

            # Get current activation levels..
            current = [node.activation_level for node in self.nodes]

            # Check for changes
            if current == prev:
                break
            
            # Has changed. Run more
            prev = current

    def derivate(self, node):
        """
            Derivative of the activation function for back-propagation.
        """
        a = node.activation_level

        if self.activation_function == Activation.sigmoid_log:
            # P(t) * (1 - P(t)) where P(t) is the logistic function
            return a * (1 - a)

        elif self.activation_function == Activation.sigmoid_tanh:
            # 1 - tanh^2 (x)
            return 1 - (Activation.sigmoid_tanh(a) ** 2)

        elif self.activation_function == Activation.step or hasattr(self.activation_function, 'func'):
            # Can't derivate descrete functions
            return 0.0        

        # Linear 
        return 1.0

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
        return self.activation_function(inpt)

    def set_learning_mode(self):
        """Reset the layer for learning."""
        self.learning_mode = True

    def set_testing_mode(self):
        """Reset the layer for testing."""
        self.learning_mode = False
    
