"""
A module for holding ANN modules.
ANN modules are complex layers and links, a collection
of layers and links who together behaves a certain way.

Module layers connect with the rest of the ANN through links,
whom can be partially created before, or by the module. In the
latter case users of a module must complete the partial links
and run setup() on them.
"""
from layer import *
from link import *


class Module(Layer):
    """A base class for complex layers with links.
    
    An ANN module is a collection of links and additional layers.
    Links which serve as upstream or downstream to the layer can be
    provided as partial links, otherwise the module will create them
    as partial links, and users must them self complete these.
    """

    def __init__(self, *args, **vargs):
        super(Module, self).__init__(*args, **vargs)
        self.layers = []
        self.links = []


class Competitive(Module):
    """
    Competitive network layers are characterized by neurons that have
    inhibitory intra-layer effects but either excitatory or inhibitory
    inter-layer influences.
    """

    def __init__(self, name, nodes, activation_function=None,
                neg=-1, pos=1, up=None, down=None, rounds=10):
        """Create a competitve network layer.

        'name' name of the new layer.
        'nodes' number of nodes or a list of nodes.
        'activation' function for the layer.
        'neg' is the negative weight for the inhibitor.
        'pos' is the positive weight for the simulator.
        'up' is optional upstream partial link.
        'down' is optional downstream partial link.
        'rounds' is the max settling rounds to achieve quiescent.
        """
        # 1) A layer of neurons using simple-linear or positive-linear
        # activation functions, so as to preserve any differences in the sum
        # of weighted inputs between neurons.
        if activation_function is None:
            activation_function = Layer.linear

        super(Competitive, self).__init__(name, nodes, activation_function)

        # 2) A link that connects the layer to itself, using a triangular
        # topology, in which all weights are fixed (no learning) and negative.
        # This achieves the inhibition by one node of all other nodes.
        self.links.append(Link(self, self, "triangulate", learning_rate=0, weights=([neg] * len(self.nodes))))

        # 3) An intra-layer link, using a 1-1 topology, with fixed
        # (no learning) positive weights. This insures that each node
        # stimulates itself during repeated updating rounds.
        self.links.append(Link(self, self, '1-1', learning_rate=0, weights=([pos] * len(self.nodes))))

        # 4) Input links from one or more upstream layers. These links will
        # often include learning, especially in cases where the neurons of
        # the competitive layer should be trained to become detectors for
        # various upstream (e.g. input) patterns.
        if up is None:
            up = Link()

        up.post_layer = self
        self.links.append(up)
        
        # 5) Output links to any downstream layers.
        if down is None:
            down = Link()

        down.pre_layer = self
        self.links.append(down)

        # Competitive layers are normally always running in quiescent
        # mode, with 10 or more settling rounds often required.
        self.quiescent_mode = True
        self.max_settling = rounds
        

class Inhibitory(Module):
    """
    The inhibitor module simply integrates all upstream excitation
    and converts it into inhibition, which it sends downstream.
    """

    def __init__(self, name, activation_function=None,
                neg=-1, pos=1, up=None, down=None):

        self.neg = neg
        self.pos = pos
        self.up = up.pre_layer
        self.down = down.post_layer

        """Create a new inhibitory module.

        'name' is the name of the inhibitory layer.
        'activation_function' is the activation function of the layer.
        'neg' is the negative weights for inhibitory arcs.
        'pos' is the positive weights for excitatory arcs.
        'up' is the optional partial upstream link.
        'down' is the optional partial downstream link.
        """
        # 1) A layer consisting of a single neuron, often using a simple
        # linear activation function.
        if activation_function is None:
            activation_function = Activation.linear

        super(Inhibitory, self).__init__(name, 1, activation_function)

        # 2) An input link coming from a layer that will be inhibiting
        # another layer (or itself). There will normally be one arc for
        # each of these upstream neurons, such that a) any one of them can
        # trigger the inhibitor, and b) the more that trigger it, the
        # stronger the total inhibition felt downstream. All arcs have the
        # same positive weight and are non-plastic.
        if up is None:
            up = Link()

        up.post_layer = self

        if up.topology is None:
            up.topology = "full"

        up.learning_rate = 0
        up.arc_range = (pos, pos)
        self.links.append(up)

        # 3) An output link going to the layer to be inhibited, with
        # normally one arc to each of its neurons. All arcs have the same
        # negative weight and are non-plastic.
        if down is None:
            down = Link()

        down.pre_layer = self
        if down.topology is None:
            down.topology = "full"

        down.learning_rate = 0
        down.arc_range = (neg, neg)
        self.links.append(down)

        # quiescent_mode and max_settling_rounds?


class Associative(Module):
    """
    Associative modules are designed for pattern storage and retrieval.
    The neurons of the the central associative layer are fully connected
    with excitatory links such that a partial pattern can be completed
    via spreading activation within the layer.
    """
    
    def __init__(self, name, nodes, rate, activation_function=None, rule=None,
            initial=(0,1), up=None, down=None, up_inh=None, rounds=10):
        """Create a new associative module.

        'name' is the name of the associative layer.
        'nodes' is number of nodes or a list of nodes for the layer.
        'rate' is the learning rate for the excitatory intra-layer link.
        'activation' is the activation function for the layer.
        'rule' is the learning rule for the excitatory intra-layer link.
        'initial' is the initial weight range for the intra-layer link.
        'up' is a optional partial upstream link to the layer.
        'down' is a optional partial downstream link to the layer.
        'up_inh' is the optional partial upstream to the inhibitory layer.
        'rounds' is the max settling rounds to achieve quiescent.
        """
        # 1) A layer of fully-connected neurons with step or sigmoidal
        # activation functions, thus insuring that, at any given time,
        # neurons can be characterized as off or on.
        if activation_function is None:
            activation_function = Activation.step

        super(Associative, self).__init__(name, nodes, activation_function)

        # 2) An intra-layer link with a full connection topology and highly
        # plastic arcs. A general Hebbian learning rule is often used.
        if rule is None:
            rule = Link.general_hebb
        self.links.append(Link(self, self, Link.FULL, 
                    rate=rate, rule=rule, initial=initial))

        # 3) An inter-layer input link from an upstream layer, Lu. These arcs
        # have high positive weights but are not plastic. The connection
        # topology is often 1-1. The high positive weights and 1-1 topology
        # enable single neurons in the upstream layer to force the
        # activation of their counterpart neuron in the associative layer,
        # thus embodying the process of loading a (complete or partial)
        # pattern from upstream layer into the associative layer.
        if up is None:
            up = Link()

        up.post_layer = self

        if up.topology is None:
            up.topology = "1-1"

        self.links.append(up)

        # 4) An inter-layer input link from an inhibitory layer/module. These
        # arcs have a high negative weight and are not plastic. This
        # manifests feedforward inhibition from Lu to the associative layer.
        if up_inh is None:
            up_inh = Link()

        down_inh = Link(post=self)

        self.inhibitor = Inhibitory("%_inh" % name, up=up_inh, down=down_inh) 
        
        self.layers.append(self.inhibitor)

        # 5) An inter-layer output link to a downstream layer that may
        # simply read out values from the associative layer or may combine
        # and further process those signals.
        if down is None:
            down = Link()

        down.pre_layer = self

        if down.topology is None:
            down.topology = '1-1'

        down.learning_rate = 0

        if down.arc_range is None:
            down.arc_range = (1, 1)

        self.links.append(down)
        
        self.max_settling = rounds
    
    def reset_for_training(self):
        """Reset the module for training."""
        super(Associative, self).reset_for_training()
        self.inhibitor.active = False
        self.quiescent_mode = False

    def reset_for_testing(self):
        """Reset the module for testing."""
        super(Associative, self).reset_for_testing()
        self.inhibitor.active = True
        self.quiescent_mode = True

    def training(self):
        self.update(self.quiescent_mode)
        # TODO:
        # After one updating round, La does learning on 
        # the intra-layer excitatory arcs.

    def testing(self):
        self.update(self.quiescent_mode)
        # TODO:
        # During recall, the inhibitory layer should be active, invoking
        # feedforward inhibition, which insures that only those neurons in
        # La that have a high positive correlation with many of the Na
        # neurons will become active. The inhibition will prevent those with
        # strong correlations to only a small subset of Na from becoming
        # active. This effectively restricts the spreading activation within
        # La to primarily those neurons that complete the partial pattern.
        # Without feedforward inhibition of this sort, the entire layer could
        # become active, since most neurons will have some positive
        # correlation to some of the Na neurons, particularly if the synapses
        # of La have been trained to store several patterns.


class Transformer(Module):
    """
    Transformer Modules are normally very simple, consisting of a single
    layer that is 1-1 connected to its immediate upstream neighbor with
    positive, non-plastic connections. Downstream connections can be of
    many types and topologies. The purpose of this layer is to transform
    the outputs of the upstream neighbor in order to pre-process them for
    the downstream neighbor.
    """

    def __init__(self, name, nodes, activation_function, up=None, down=None):
        """Create a transformer layer.
        
        'name' is the name of the transformer layer.
        'nodes' is number of nodes or a list of nodes for the layer.
        'activation_function' is the activation function for the layer.
        'up' is an optional partial upstream link to the layer.
        'down' is an optional partial downstream link to the layer.
        """
        super(Transformer, self).__init__(name, nodes, activation_function)
        
        # Upstream link
        if up is None:
            up = Link()

        up.post_layer = self

        up.topology = '1-1'
        up.learning_rate = 0

        if up.arc_range is None:
            up.arc_range = (1, 1)

        self.links.append(up)

        # Downstream link
        if down is None:
            down = Link()

        down.pre_layer = self

        self.links.append(down)

