import random

from functools import partial

from controllers.webann.ann.layer import *
from controllers.webann.ann.link import *
from controllers.webann.ann.ann import Ann
from controllers.webann.ann.parser import AnnParser
from controllers.webann.ann.ann_modules import Inhibitory

import time


data = [
    [[1, 1], [0]],
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

# Layers
i_l = Layer("Input", 2, io_type='input')
hidden = Layer("Hidden", 2, partial(Activation.step, T=2))
out = Layer("Out", 1, partial(Activation.step, T=2), io_type='output')

layers = [i_l, hidden, out]
#     weights=[2, -1, -1, 2],

l1 = Link(i_l, hidden, 
    arc_range=[-1, 2],
    arcs=[(0,0), (0,1), (1,0), (1,1)],
    learning_rule=LearningRule.hebbian
    )

l2 = Link(hidden, out, 
    arc_range=[0, 2],
    topology="full",
    learning_rule=LearningRule.general_hebb
    )

# Execution order
ann = Ann(layers, [l1, l2])
ann.execution_order = layers


# Do training
ann.reset_for_training()

epochs = 1000
# Learn

# Run back-propagation learning
t = time.time()
print "Performing %i epochs of back propagation learning" % epochs
for i in range(epochs):
    inputs, target = data[i % len(data)]
    ann.training(inputs)
    print l1.export_weights()
    print l2.export_weights()

print "Finished in %.2f secs, ANN is ready" % (time.time() - t)

print l1.export_weights()
print l2.export_weights()

# Do testing
ann.reset_for_testing()

print l1.export_weights()
print l2.export_weights()


# AnnParser.export(ann, "scripts/xor.ini")

print ann.recall([0, 0])
print ann.recall([1, 0])
print ann.recall([1, 1])
print ann.recall([0, 1])

