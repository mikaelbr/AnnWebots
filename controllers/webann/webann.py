## This uses the EpuckBasic code as the interface to webots, and the epuck2 code to connect an ANN
# to webots.

import epuck_basic as epb
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

# The webann is a descendent of the webot "controller" class, and it has the ANN as an attribute.
class WebAnn(epb.EpuckBasic):

    def __init__(self, ann, tempo = 1.0):
        epb.EpuckBasic.__init__(self)

        self.basic_setup() # defined for EpuckBasic 

        self.ann = ann
        self.ann.init_nodes()

        self.tempo = tempo   

    def drive_speed(self, left=0, right=0):
        """
            Drive with speed X and Y for left wheel (X) and right wheel (Y)
        """
        ms = self.tempo * 1000
        self.setSpeed(int(left * ms), int(right * ms))

    def run(self):
        
        self.forward(1)
        self.spin_angle(180)

        while True: # main loop
            dist = [max(-1, (1 - (i / 600))) for i in self.get_proximities()]
            cam = process_snapshot(self.snapshot(),color="green")
            inputs = dist + cam

            print "Distance"
            print dist

            print "Camera"
            print cam

            self.drive_speed(*self.ann.recall(inputs))

            if self.step(self.timestep) == -1: break


class Static(WebAnn):

    def __init__(self, ann, tempo = 1.0):

        super(Static, self).__init__(ann, tempo)
        self.ann.reset_for_training()


class Learning(WebAnn):
    """A robot with backprop learning from a datafile."""

    def __init__(self, ann, learn=True):

        super(Learning, self).__init__(ann)

        self.ann.reset_for_training()

        if learn:
            self.learning('data/learning.txt', epochs=5000)
            self.ann.reset_for_testing()

    def learning(self, data_file, epochs=1):
        # Read inputs and targets from datafile
        self.data = []

        with open(data_file, 'r') as f:
            for line in f.readlines():
                self.data.append(eval(line))

        # Add forward cases
        tmp = [[1]*(self.num_dist_sensors) + [0]*5, (1, 1)]
        self.data.extend([tmp for i in range(len(self.data)/4)])

        # Run back-propagation learning
        t = time.time()
        print "Performing %i epochs of back propagation learning" % epochs
        for i in range(epochs):
            inputs, target = self.data[i % len(self.data)]
            self.ann.back_propagation(inputs, target)

        print "Finished in %.2f secs, ANN is ready" % (time.time() - t)




# ann = AnnParser("ann/scripts/static.ini").create_ann()
# controller = Static(ann, tempo = 1.0)

ann = AnnParser("ann/scripts/learning.ini").create_ann()
controller = Learning(ann)


controller.run()
