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


    def init_nodes(self):

        if self.initz: # Avoid multiple initializations
            return
        self.initz = True

        # Find nested layers and links
        for i in self.layers + self.links:

            if hasattr(i, 'links'):
                for link in i.links:
                    if link not in self.links:
                        self.links.append(link)


            if hasattr(i, 'layers'):
                for layer in i.layers:
                    if layer not in self.layers:
                        self.layers.append(layer)


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

        self.layer_mapped = layer_order

        self.execution_order = [layer_order[str(layer).lower()] 
                        for i, layer in enumerate(self.execution_order)]

        for link in self.links:
            link.generate_arcs()

        # Fill up layers not in execution order list. 
        if len(self.execution_order) < len(self.layers):
            self.execution_order.extend([l for l in self.layers if l not in self.execution_order])


        # Find link learning order for back propagation
        all_arcs = [] # To make sure each arc is visited once
        for link in self.links:
            all_arcs.extend(link.arcs)
        
        current = self.output_nodes
        distance = 0 # Distance from output layers
        order = {} # Links and their largest distance
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

        items = sorted(order.items(), key=itemgetter(1))
        self.learning_order = [l for l, i in items]

        # Is it possible for nodes to not contribute to output?
        for link in self.links:
            if link not in self.learning_order:
                self.learning_order.append(link)


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

    def training(self, inputs):
        """
            Perform one epoch of unsupervised training.
        """

        self.init_nodes()
        self.set_inputs(inputs)

        for layer in self.execution_order:
            layer.update()

        for link in self.learning_order:
            link.learn()

        return self.get_output()

    def back_propagation(self, inputs, targets):
        """Perform one epoch of incremental back propagation learning."""
        self.init_nodes()
        self.set_inputs(inputs)

        for layer in self.execution_order:
            layer.update()

        for link in self.learning_order:
            link.back_propagation(targets, self.output_nodes)

        return self.get_output()

    def testing(self, inputs, targets):
        """
            Test without learning the inputs, returns the error.
        """

        wheels = self.recall(inputs)

        return sum([(targets[i] - wheels[i])**2 for i in range(len(targets))])

    def reset_for_training(self):
        """Reset the ANN for training."""
        for i in self.layers:
            i.reset_for_training()

    def reset_for_testing(self):
        """Reset the ANN for testing."""
        for i in self.layers:
            i.reset_for_testing()


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
