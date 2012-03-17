from layer import *
from arc import *
from node import *
from link import *
from functools import partial


class Ann(object):

    def __init__(self, layers, links, execution_order = []):

        self.layers = layers
        self.links = links

        self.execution_order = execution_order
        self.initz = False


    def init_nodes(self):

        if self.initz: # Avoid multiple initializations
            return
        self.initz = True

        # Find input and output
        self.input_nodes = []
        self.output_nodes = []

        for layer in self.layers:
            if layer.type is None:
                continue

            if layer.type.lower() == "input":
                self.input_nodes.extend(layer.nodes)

            elif layer.type.lower() == "output":
                self.output_nodes.extend(layer.nodes)


        # Fix execution order. 
        layer_order = {layer.name.lower(): layer 
                        for layer in self.layers}
        self.execution_order = [layer_order[str(layer).lower()] 
                        for i, layer in enumerate(self.execution_order)]

        # Fill up layers not in execution order list. 
        if len(self.execution_order) < len(self.layers):
            self.execution_order.extend([l for l in self.layers if l not in self.execution_order])


    def recall(self, inputs):
        """
            Sets input and update all nodes

            Returns output node values.
        """
        self.init_nodes()
        self.set_inputs(inputs)

        for layer in self.execution_order:
            layer.update()

        return self.get_output()


    def set_inputs(self, inputs):
        """
            Set activation level for input nodes.
        """
        for i, node in enumerate(self.input_nodes):
            node.activation_level = inputs[i]

    def get_output(self):
        return [n.activation_level for n in self.output_nodes]


if __name__ == "__main__":
    dist = Layer("Distance", 8, io_type="input")
    cam = Layer("Camera", 5, io_type="input")
    obstr = Layer("Obstructions", 2, partial(Activation.step, threshold=0))
    actions = Layer("Actions", 4, partial(Activation.step, threshold=0.6))
    wheels = Layer("Speed", 2, Activation.sigmoid_tanh, io_type="output")

    distance = Layer("Distance", 6, io_type="input")
    camera = Layer("Camera", 5, io_type="input")
    hidden = Layer("Hidden", 11)
    out = Layer("Output", 2, io_type="output")

    layers = [distance, camera, hidden, out]

    # dh_links = Link(distance, hidden, "1-1")
    # ch_links = Link(camera, hidden, "1-1")
    # ho_links = Link(hidden, out, "full")
    dh_links = Link(distance, hidden, weights=(0.5, 0.5, -1, -1, 0.4, 0.4), arcs=[(0, 2), (1, 1), (0, 0), (1, 0), (0, 3), (1, 3)])
    ch_links = Link(camera, hidden, weights=(0.5, 0.5, -1, -1, 0.4), arcs=[(0, 2), (1, 1), (0, 0), (1, 0), (0, 3)])
    ho_links = Link(hidden, out, "full", weights=[1.3, 1.3, -0.5, 0.5, 0.6, -0.6, -0.3, -0.5])

    gann = Ann(layers, [dh_links, ch_links, ho_links])
    gann.execution_order = layers

    import random
    def ra():
        return random.random()

    print gann.recall([ra() for i in range(len(hidden.nodes))])
    print gann.recall([ra() for i in range(len(hidden.nodes))])
    print gann.recall([ra() for i in range(len(hidden.nodes))])
    print gann.recall([ra() for i in range(len(hidden.nodes))])
