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

# AnnParser.export(ann, "ann/scripts/static.ini")

# ann = AnnParser("ann/scripts/static.ini").create_ann()
controller = Static(ann, tempo = 1.0)
controller.run()
