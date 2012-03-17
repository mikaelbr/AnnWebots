## This uses the EpuckBasic code as the interface to webots, and the epuck2 code to connect an ANN
# to webots.

import epuck_basic as epb
#import graph
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

# The webann is a descendent of the webot "controller" class, and it has the ANN as an attribute.

class WebAnn(epb.EpuckBasic):

    def __init__(self, ann, tempo = 1.0):
        epb.EpuckBasic.__init__(self)

        self.basic_setup() # defined for EpuckBasic 

        self.ann = ann
        self.ann.init_nodes()

        self.tempo = tempo

    def scale_proximities(self, distances):
        """Scale distance values between -1 and 1."""
        return [max(-1, (1 - (i / 600))) for i in distances]
    
    def run(self):

        while True: # main loop

            # Get input proximity and camera.
            dist = (self.get_proximities())
            print dist

            d = self.scale_proximities(dist)
            print d
            # For testing, set random values.
            i = d + [random.random() for x in range(5)]

            left, right = self.ann.recall(i)

            self.move_wheels(left, right, self.tempo)
            print process_snapshot(self.snapshot(),color="green")
            if self.step(64) == -1: break


    def long_run(self,steps = 1000):
        #self.ann.simsteps = steps
        #self.spin_angle(prims1.randab(0,360))
        #
        self.spin_angle(-70)
        self.stop_moving()
        self.backward()
        self.run_timestep(200)
        #self.ann.redman_run()
        image = self.snapshot()
        # print "avg_rgb " , avg_rgb(image)

        #img = Image.open()
        #print "lengde ", len(column_avg(img))
        # print
        #print column_avg(image)

#*** MAIN ***
# Webots expects a controller to be created and activated at the bottom of the controller file.

# Layers
dist = Layer("Distance", 8, io_type='input')
cam = Layer("Camera", 5, io_type='input')
obstr = Layer("Obstructions", 2, partial(Activation.step, T=0))
actions = Layer("Actions", 4, partial(Activation.step, T=0.6))
wheels = Layer("Speed", 2, Activation.sigmoid_tanh, "output")
stop = Inhibitory("Stop", activation_function=Activation.step, neg=-4, pos=0.1, up=Link(pre_layer=cam), down=Link(post_layer=actions))

layers = [dist, cam, obstr, stop, actions, wheels]

# Obstruction upstream link
o_up = Link(dist, obstr, weights=(-1, -1, -1, -1),
            arcs=[(6, 0), (7, 0), (0, 1), (1, 1)])

# Obstruction downstream link
o_down = Link(obstr, actions, weights=(0.5, 0.5, -1, -1, 0.4, 0.4),
            arcs=[(0, 2), (1, 1), (0, 0), (1, 0), (0, 3), (1, 3)])

# Distance downstream link
# 0=NNE, 1=NE, 2=E, 3=SE, 4=SW, 5=W, 6=NW, 7=NNW
d_down = Link(dist, actions,
            weights=(0.5, 0.8, 0.8, 0.5, 0.1, 0.25, 0.25, 0.1, 0.25, 0.25),
            arcs=[(1,0),(0,0),(7,0),(6,0),(6,1),(5,1),(4,1),(1,2),(2,2),(3,2)])

# Camera downstream link
c_down = Link(cam, actions, weights=(0.5, 0.5, 0.1, 0.5, 0.5, -1),
            arcs=[(0, 1), (1, 1), (2, 0), (3, 2), (4, 2), (2, 3)])

# Obstruction -> Stop link
s_up = Link(obstr, stop, "full", weights=(0.2, 0.2))

# Wheels upstream link
w_up = Link(actions, wheels, "full",
            weights=(1.3, 1.3, -0.5, 0.5, 0.6, -0.6, -0.3, -0.5))

# Execution order
ann = Ann(layers, [s_up, o_up, o_down, d_down, c_down, w_up])
ann.execution_order = layers

# ini_parse = AnnParser("ann/scripts/distance.ini")
# gann = ini_parse.create_ann()

controller = WebAnn(ann, tempo = 1.0)
controller.run()

# controller.long_run(40)
#controller.run_toy()