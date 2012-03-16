# This uses the EpuckBasic code as the interface to webots, and the epuck2 code to connect an ANN
# to webots.

import epuck2
import epuck_basic as epb
#import graph
import prims1
from imagepro import *

# The webann is a descendent of the webot "controller" class, and it has the ANN as an attribute.

class WebAnn(epb.EpuckBasic):

  def __init__(self, tempo = 1.0, e_thresh = 125, nvect = True, cvect = True, svect = True, band = 'bw', concol = 1.0, snapshow = True, ann_cycles = 1, agent_cycles = 5, act_noise = 0.1, tfile = "redman4"):
    epb.EpuckBasic.__init__(self)
    self.basic_setup() # defined for EpuckBasic 
    self.ann = epuck2.annpuck2(agent = self, e_thresh = e_thresh, nvect = nvect, cvect = cvect, svect = svect, band = band, snapshow = snapshow,
                   concol = concol, ann_cycles = ann_cycles, agent_cycles = agent_cycles, act_noise = act_noise,
                   tfile = tfile)
    


  def long_run(self,steps = 500):
      self.ann.simsteps = steps
      self.spin_angle(prims1.randab(0,360))
      #self.backward()
      self.run_timestep(200)
      self.ann.redman_run()
      image = self.snapshot()
      img_avg= image_avg(image)
      print "avg ", image_avg

#*** MAIN ***
# Webots expects a controller to be created and activated at the bottom of the controller file.

controller = WebAnn(tempo = 1.0, band = 'gray')
controller.long_run(40)
#controller.run_toy()