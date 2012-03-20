from layer import *
from arc import *
from node import *
from link import *
from functools import partial
from operator import itemgetter


class Ann(object):

    def __init__(self, layers, links, execution_order = []):

        self.layers = layers
        self.links = links

        self.execution_order = execution_order
        self.initz = False


    def extract_execution_order(self):
         # Fix execution order. 
        layer_order = {layer.name.lower(): layer for layer in self.layers}
        # Replace with objects if text. (Also works with objects)
        self.execution_order = [layer_order[str(layer).lower()] for i, layer in enumerate(self.execution_order)]

        # Fill up layers not in execution order list. 
        if len(self.execution_order) < len(self.layers):
            self.execution_order.extend([l for l in self.layers if l not in self.execution_order])


    def create_encoders_decoders(self):
        # Find input and output
        self.input_nodes = []
        self.output_nodes = []

        for layer in self.layers:
            if layer.type is None:
                continue

            if layer.type.lower() == "encoder":
                self.input_nodes.extend(layer.nodes)

            elif layer.type.lower() == "decoder":
                self.output_nodes.extend(layer.nodes)

    def init_nodes(self):

        if self.initz: # Avoid multiple initializations
            return
        self.initz = True

        self.append_intra_layers()

        self.create_encoders_decoders()
        
        self.extract_execution_order()

        # Generate all arc weights. 
        for link in self.links:
            link.generate_arcs()

        self.find_link_order()

    def append_intra_layers(self):
        """
            Append intra layer and links. 
        """
        for i in self.layers + self.links:

            if hasattr(i, 'layers'):
                for layer in i.layers:
                    if layer not in self.layers:
                        self.layers.append(layer)

            if hasattr(i, 'links'):
                for link in i.links:
                    if link not in self.links:
                        self.links.append(link)


    def find_link_order(self):
        """
         Use distance to find the order of the links.
         Used by back propagation. Should be ordered like
         L2 -> L3, L1 -> L2, L0 -> L1
         for a network with topology L0 -> L1 -> L2 -> L3
        """
        all_arcs = []
        for l in self.links:
            all_arcs.extend(l.arcs)

        current = self.output_nodes
        distance = 0
        order = {}
        while all_arcs and current:
            next_ = []

            for node in current:

                for link in node.layer.entering:

                    for arc in link.arcs:
                        if arc in all_arcs:
                            all_arcs.remove(arc)
                            next_.append(arc.pre_node)
                            order[link] = distance


            current = next_
            distance += 1

        # Sort by distance
        items = sorted(order.items(), key=itemgetter(1))
        self.link_order_learn = [l for l, i in items]

        # Append other nodes.
        for link in self.links:
            if link not in self.link_order_learn:
                self.link_order_learn.append(link)



    def recall(self, inputs):
        """
            Sets input and update all nodes

            Returns output node values.
        """
        self.init_nodes()
        self.set_input(inputs)

        for layer in self.execution_order:
            layer.update()

        return self.get_result()

    def reset(self):
        """
            Reset all nodes and arcs in the ANN.
        """
        for layer in self.layers:
            for node in layer.nodes:
                node.reset_levels()

        for link in self.links:
            for arc in link.arcs:
                arc.reset()

    def learn(self, inputs):
        """
            Do one iteration/epoch of learning,
            no back propagation, using learning rules
        """

        self.recall(inputs)

        for link in self.link_order_learn:
            link.learn()

        return self.get_result()

    def backprop(self, inputs, targets):
        """Perform one epoch of incremental back propagation learning."""
        self.recall(inputs)

        for link in self.link_order_learn:
            link.backprop(targets, self.output_nodes)

        return self.get_result()

    def test(self, inputs, targets):
        """
            Test without learning the inputs.
            Returns error.
        """

        output = self.recall(inputs)

        return sum([(targets[i] - output[i])**2 for i in range(len(targets))])

    def set_learning_mode(self):
        """
            Set all layers to learning mode. 
        """
        for i in self.layers:
            i.set_learning_mode()

    def set_testing_mode(self):
        """
            No learning. only do testing. 
        """
        for i in self.layers:
            i.set_testing_mode()


    def set_input(self, inputs):
        """
            Set activation level for input nodes.
        """
        for i, node in enumerate(self.input_nodes):
            node.activation_level = inputs[i]

    def get_result(self):
        return [n.activation_level for n in self.output_nodes]