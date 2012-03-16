

class Node(object):
    """
        The ANN node class, holding the nodes current and previous activation level,
        and the membrane potential.

        Is a part of a layer, and uses that layers activation function to calculate
        the activation level.
    """

    def __init__(self, layer):
        self.layer = layer # Set node to be a part of a layer

        self.reset_levels()

        self.encoders = [] # input arcs
        self.decoders = [] # output arcs

    def reset_levels(self):
        """
            Reset all levels
        """
        this.membrane_potential = 0
        this._activation_level = 0 # bypass the property func.
        this.prev_activation_level = 0

    @property
    def activation_level(self):
        return self._activation_level

    @activation_level.setter
    def activation_level(self, value):
        """
            Used to remember the previous activation level
        """
        self.prev_activation_level = self._activation_level
        self._activation_level = value

    def activate(self):
        """
            Updates the activation level based on the layers
            activation function. Calculates the weighted input and
            passes it to the layer.
        """
        weighted_input = 0

        for arc in self.encoders:
            con_node = 0 # arc.from # connecting node
            if not con_node.layer.active:
                continue

            if con_node.layer is self.layer:
                weighted_input += con_node.prev_activation_level * arc.weight
            else:
                weighted_input += con_node.activation_level * arc.weight 

        self.membrane_potential = weighted_input
        self.activation_level = self.layer.activation_function(self.membrane_potential)



        