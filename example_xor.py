import random

from functools import partial

from controllers.webann.ann.layer import *
from controllers.webann.ann.link import *
from controllers.webann.ann.ann import Ann
from controllers.webann.ann.parser import AnnParser
from controllers.webann.ann.ann_modules import Inhibitory

parser = AnnParser("example_scripts/xor.ini")
ann = parser.create_ann()

# # Layers
# i_l = Layer("Input", 2, io_type='input')
# hidden = Layer("Hidden", 2, partial(Activation.step, T=2))
# out = Layer("Out", 1, partial(Activation.step, T=2), io_type='output')

# layers = [i_l, hidden, out]

# l1 = Link(i_l, hidden, 
#     weights=[2, -1, -1, 2],
#     arcs=[(0,0), (0,1), (1,0), (1,1)]
#     )

# l2 = Link(hidden, out, 
#     weights=[2, 2],
#     topology="full"
#     )

# # Execution order
# ann = Ann(layers, [l1, l2])
# ann.execution_order = layers


# AnnParser.export(ann, "scripts/xor.ini")

print ann.recall([0, 0])
print ann.recall([1, 0])
print ann.recall([1, 1])
print ann.recall([0, 1])

