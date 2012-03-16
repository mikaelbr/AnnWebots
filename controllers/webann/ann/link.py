import random
from arc import *
from node import *
from layer import *


class Link(object):

    def __init__(self,pre_layer,post_layer,topology="full",arc_range_min=-1.0, arc_range_max=1.0,learning_rate=0.2,learning_rule = None):
        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.topology = topology
        self.arc_range_max = arc_range_max
        self.arc_range_min = arc_range_min
        self.learning_rate = learning_rate
        self.learning_rule = learning_rule

        self.generate_arcs()

    def get_random_weight(self):
        return random.uniform(self.arc_range_min, self.arc_range_max)

    def generate_arcs(self, connection_prob=0.4):
        """
        Generates the arcs based on the connection topology type
        Node(pre_node,post_node,current_weight,init_weight,link)
         self.arcs = [Arc(Node(),Node(),self.get_random_weight(),self.get_random_weight(),self) for i in range(n_arcs)]
        """
        self.arcs = []

        pre_nodes = self.pre_layer.nodes
        post_nodes = self.post_layer.nodes

        def connect(pre_node, post_node):
            new_arc = Arc(pre_node, post_node, self.get_random_weight(), self)
            self.arcs.append(new_arc)
            pre_node.outgoing.append(new_arc)
            post_node.incomming.append(new_arc)


        if self.topology == '1-1':
            [[connect(pre_node, post_node) for j, pre_node in enumerate(pre_nodes) if i == j ] for i, post_node in enumerate(post_nodes)]

        elif self.topology == 'full':
            [[connect(pre_node, post_node) for pre_node in (pre_nodes)] for post_node in (post_nodes)]

        elif self.topology == 'stochastic':
            [[connect(pre_node, post_node) for pre_node in (pre_nodes)  if random.random() < connection_prob] for post_node in (post_nodes)]

        elif self.topology == 'triangulate':
            [[connect(pre_node, post_node) for j, pre_node in enumerate(pre_nodes) if i != j] for i, post_node in enumerate(post_nodes)]


        self.pre_layer.exiting = self.arcs
        self.post_layer.entering = self.arcs

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
