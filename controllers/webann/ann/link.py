import random
from arc import *
from node import *
from layer import *

class LearningRule(object):

    @staticmethod
    def hebbian(arc, rate, pre, post):
        return rate * pre * post

    @staticmethod
    def general_hebb(arc, rate, pre, post, threshold=0.5):
        return rate * (pre - threshold) * (post - threshold)

    @staticmethod
    def oja(arc, rate, pre, post):
        return rate * pre * (post - (pre * arc.current_weight))

class Link(object):

    def __init__(self, pre_layer = None, post_layer = None, topology=None, 
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

        elif self.topology == '1-1' or not(self.topology):
            self.arcs = [connect(pre_node, post_node) for j, pre_node in enumerate(pre_nodes) for i, post_node in enumerate(post_nodes) if i == j]

        elif self.topology == 'full':
            self.arcs = [connect(pre_node, post_node) for pre_node in (pre_nodes) for post_node in (post_nodes)]

        elif self.topology == 'stochastic':
            self.arcs = [connect(pre_node, post_node) for pre_node in (pre_nodes) for post_node in (post_nodes) if random.random() < connection_prob]

        elif self.topology == 'triangulate':
            self.arcs = [connect(pre_node, post_node) for j, pre_node in enumerate(pre_nodes) for i, post_node in enumerate(post_nodes) if i != j]

        elif self.topology == '2-1':
            lena, lenb = len(pre_nodes), len(post_nodes)
            self.arcs = [connect(pre_nodes[i % lena], post_nodes[(i+j) % lenb] ) for j in range(2) for i in range(max(lena, lenb))]


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

    def learn(self):
        """
            Use the learning rule to alter weights
            of arcs. 
        """        
        
        if self.learning_rate and self.post_layer.learning_mode:
            for arc in self.arcs:
                arc.current_weight += self.learning_rule(arc, self.learning_rate,
                        arc.pre_node.activation_level,
                        arc.post_node.activation_level)

    def backprop(self, targets, outputs):
        """
            Using back propagation 

            1. Instruct the link's post-synaptic layer to update 
                its delta values.

            2. Propagate these post-synaptic deltas (multiplied by 
                the respective arc weights) to the pre-synaptic layer.

            3. Modify the arc weights using the post-synaptic delta values.

        """

        # 1.
        for node in self.post_layer.nodes:

            if node.layer.type and node.layer.type.lower() == "decoder":
                delta = targets[outputs.index(node)] - node.activation_level  
            else: 
                delta = node._delta

            node._delta = node.layer.derivate(node) * delta

        # 2.
        for node in self.pre_layer.nodes:
            delta = sum([arc.current_weight * arc.post_node._delta for arc in node.outgoing ])

            node._delta = delta

        # 3.
        diff_val = lambda a: (self.learning_rate * a.pre_node.activation_level * a.post_node._delta)
        for arc in self.arcs:
            arc.current_weight += diff_val(arc)

