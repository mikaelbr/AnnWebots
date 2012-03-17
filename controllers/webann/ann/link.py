import random
from arc import *
from node import *
from layer import *


class Link(object):

    def __init__(self, pre_layer, post_layer, topology=None, 
                arc_range=[-0.1, 0.1], learning_rate=0.2, 
                weights=None, arcs=None,
                learning_rule = None):

        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.topology = topology
        self.arc_range = arc_range
        self.learning_rate = learning_rate
        self.learning_rule = learning_rule

        self.arcs = arcs
        self.init_arcs = arcs # For using in export

        self.weights = weights
        self.init_weights = weights # for using in export

        self.generate_arcs()

    def get_random_weight(self):
        return random.uniform(*self.arc_range)

    def export_arcs(self):
        arcs = []

        pre_nodes = self.pre_layer.nodes
        post_nodes = self.post_layer.nodes

        for i, arc in enumerate(self.arcs):
            arcs.append( (pre_nodes.index(arc.pre_node), post_nodes.index(arc.post_node)) )

        return arcs

    def export_weights(self):
        weights = []

        for i, arc in enumerate(self.arcs):
            weights.append(arc.current_weight)

        return weights


    def generate_arcs(self, connection_prob=0.4):
        """
        Generates the arcs based on the connection topology type
        Node(pre_node,post_node,current_weight,init_weight,link)
         self.arcs = [Arc(Node(),Node(),self.get_random_weight(),self.get_random_weight(),self) for i in range(n_arcs)]
        """

        pre_nodes = self.pre_layer.nodes
        post_nodes = self.post_layer.nodes
            
        connect = lambda pre_node, post_node: Arc(pre_node, post_node, link=self)

        if self.arcs:
            self.arcs = [connect(pre_nodes[i], post_nodes[j]) for i, j in self.arcs]

        elif self.topology == '1-1':
            self.arcs = [connect(pre_node, post_node) for j, pre_node in enumerate(pre_nodes) for i, post_node in enumerate(post_nodes) if i == j]

        elif self.topology == 'full':
            self.arcs = [connect(pre_node, post_node) for pre_node in (pre_nodes) for post_node in (post_nodes)]

        elif self.topology == 'stochastic':
            self.arcs = [connect(pre_node, post_node) for pre_node in (pre_nodes) for post_node in (post_nodes) if random.random() < connection_prob]

        elif self.topology == 'triangulate':
            self.arcs = [connect(pre_node, post_node) for j, pre_node in enumerate(pre_nodes) for i, post_node in enumerate(post_nodes) if i != j]


        self.pre_layer.exiting.append(self)
        self.post_layer.entering.append(self)

        # Add weights to the arcs
        for i, arc in enumerate(self.arcs):
            if self.weights is not None and len(self.weights) > i:
                weight = self.weights[i]
            else:
                weight = random.uniform(*self.arc_range)

            arc.current_weight = weight
            arc.init_weight = weight

        return self.arcs

def main():
    print '___TESTING___'
    post_nodes = [i for i in range(6)]
    pre_nodes = [i for i in range(6)]
    link = Link("layer 1","Layer 2","stochastic",0,100,0.5,"back prop")
    arcs = link.generate_arcs(post_nodes,pre_nodes,0.1)
    print len(arcs)
if __name__ == '__main__':
    main()
