learning_rate = 0.3
dist = Layer("Distance", 8, io_type="input")
cam = Layer("Camera", 5, io_type="input")
hidden = Layer("Hidden", 13, Activation.sigmoid_tanh)
wheels = Layer("Speed", 2, Activation.sigmoid_tanh, io_type="output")

# Links
one = Link(dist, hidden, '2-1', learning_rule=LearningRule.general_hebb, learning_rate=learning_rate, arc_range=(-0.1, 0.1))
two = Link(cam, hidden, '2-1', learning_rule=LearningRule.general_hebb, learning_rate=learning_rate, arc_range=(-0.1, 0.1))
three = Link(hidden, wheels, 'full', learning_rule=LearningRule.general_hebb, learning_rate=learning_rate, arc_range=(-0.1, 0.1))

layers = [dist, cam, hidden, wheels]
# Execution order
ann = Ann(layers, [one, two, three])
ann.execution_order = layers

AnnParser.export(ann, "ann/scripts/learning.ini")