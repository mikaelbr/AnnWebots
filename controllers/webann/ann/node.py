from arc import Arc

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

        self.incomming = [] # input arcs
        self.outgoing = [] # output arcs

    def reset_levels(self):
        """
            Reset all levels
        """
        self.membrane_potential = 0
        self._activation_level = 0 # bypass the property func.
        self.prev_activation_level = 0

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
        if not self.incomming:
            return

        weighted_input = 0


        for arc in self.incomming:
            con_node = arc.pre_node # connecting node
            if not con_node.layer.active:
                continue

            if con_node.layer is self.layer:
                weighted_input += con_node.prev_activation_level * arc.current_weight
            else:
                weighted_input += con_node.activation_level * arc.current_weight 

        self.membrane_potential = weighted_input
        # print self.membrane_potential
        self.activation_level = self.layer.activation_function(self.membrane_potential)



        
