from layer import *
from arc import *
from node import *
from link import *

class Ann(object):

    def __init__(self, layers, links, execution_order = []):

        self.layers = layers
        self.links = links

        self.execution_order = execution_order

        
    def init_nodes(self):

       

        layers_map = {} # Mapping with layer name to layer
        for layer in self.layers:
            layers_map[layer.name] = layer

        # Find input and output
        self.input_nodes = []
        self.output_nodes = []

        for layer in self.layers:
            if layer.type is None:
                continue

            if layer.type.lowercase() == "input":
                self.input_nodes.extend(layer.nodes)

            elif layer.lowercase() == "input":
                self.output_nodes.extend(layer.nodes)

        # Replace name or index with actual layer in execution order
        for i, layer in enumerate(self.execution_order):
            self.execution_order[i] = layers_map[layer]


        if len(self.execution_order) < len(self.layers):
            rest = [l for l in self.layers if l not in self.execution_order]
            self.execution_order.extend(rest)





    def update(self):

        for l in self.layers:
            l.update()


    def set_inputs(self, inputs):
        """Set activation level for input nodes."""
        for i, node in enumerate(self.input_nodes):
            node.activation_level = inputs[i]

    def get_output(self):
        return [n.activation_level for n in self.output_nodes]


input_layer = Layer("Input", 4, io_type="input")
hidden_layer = Layer("Hidden", 4)
output_layer = Layer("Output", 2, io_type="output")


ih_links = Link(input_layer, hidden_layer, "stochastic")
ho_links = Link(hidden_layer, output_layer, "full")

gann = Ann([input_layer, hidden_layer, output_layer])

gann.set_inputs([1, 0, -1, 0])

gann.update()

print gann.get_output()