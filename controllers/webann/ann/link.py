
from arc import *
from node import *
from layer import *
import random

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
        return random.randrange(self.arc_range_min, self.arc_range_max, 2)

    return pre_node,post_node

    def generate_arcs(self):
        """
        Node(pre_node,post_node,current_weight,init_weight,link)

        """
        if(
        self.arcs = [Arc(Node(),Node(),self.get_random_weight(),self.get_random_weight(),self) for i in range(n_arcs)]
        return self.arcs

def main():
    print '___TESTING___'
    nodes = [Node() for i in range(20)]

    link = Link(Layer(),Layer(),"full",0,100,0.5,"back prop")
    arcs = link.generate_arcs(10)
    for arc in arcs:
        print "current weight" , arc.current_weight

if __name__ == '__main__':
    main()