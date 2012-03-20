import prims1
from imagepro import *
import Image

import random

from functools import partial

from ann.layer import *
from ann.link import *
from ann.ann import Ann
from ann.parser import AnnParser
from ann.ann_modules import Inhibitory

import time

# Do training

ann = AnnParser("ann/scripts/learning.ini").create_ann()

ann.reset_for_training()

epochs = 60
# Learn

# Read inputs and targets from datafile
data = []

with open('data/learning.txt', 'r') as f:
    for line in f.readlines():
        data.append(eval(line))

# Run back-propagation learning
t = time.time()
print "Performing %i epochs of back propagation learning" % epochs
for i in range(epochs):
    inputs, target = data[i % len(data)]
    ann.back_propagation(inputs, target)
    print [a.current_weight for a in ann.output_nodes[0].incomming]

print "Finished in %.2f secs, ANN is ready" % (time.time() - t)

print [a.current_weight for a in ann.output_nodes[0].incomming]

# Do testing
ann.reset_for_testing()
