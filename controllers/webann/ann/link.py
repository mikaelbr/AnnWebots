import random
from arc import *
from node import *
from layer import *


class Link(object):

    def __init__(self,pre_layer,post_layer,topology,arc_range_min,arc_range_max,learning_rate,learning_rule):
        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.topology = topology
        self.arc_range_max = arc_range_max
        self.arc_range_min = arc_range_min
        self.learning_rate = learning_rate
        self.learning_rule = learning_rule

    def get_random_weight(self):
        return random.uniform(self.arc_range_min, self.arc_range_max)

    def generate_arcs(self,pre_nodes,post_nodes,connection_prob=0.2):
        """
        Generates the arcs based on the connection topology type
        Node(pre_node,post_node,current_weight,init_weight,link)
         self.arcs = [Arc(Node(),Node(),self.get_random_weight(),self.get_random_weight(),self) for i in range(n_arcs)]
        """
        self.arcs = []
        add = lambda pre_node, post_node: self.arcs.append(Arc(pre_node,post_node,self.get_random_weight(),self.get_random_weight(),self))
        if self.topology == '1-1':
            [[add(pre_node, post_node) for j, pre_node in enumerate(pre_nodes) if i == j ] for i, post_node in enumerate(post_nodes)]

        elif self.topology == 'full':
            [[add(pre_node, post_node) for pre_node in (pre_nodes)] for post_node in (post_nodes)]

        elif self.topology == 'stochastic':
            [[add(pre_node, post_node) for pre_node in (pre_nodes)  if random.random() < connection_prob] for post_node in (post_nodes)]

        elif self.topology == 'triangulate':
            [[add(pre_node, post_node) for j, pre_node in enumerate(pre_nodes) if i != j] for i, post_node in enumerate(post_nodes)]
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
