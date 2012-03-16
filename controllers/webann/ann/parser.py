import ConfigParser
import pprint
from functools import wraps, partial
from layer import *
from link import *
from ann import Ann


funcs = {
    'log': Activation.sigmoid_log,
    'tanh': Activation.sigmoid_tanh,
    'step_0': partial(Activation.step, T=0),
    'step_5': partial(Activation.step, T=0.5),
    'step_8': partial(Activation.step, T=0.8),
    'linear': Activation.linear,
    'pos_linear': Activation.pos_linear
}

def fail(fn):

    @wraps(fn)
    def wrapper_fn(self, section, key, default):
        try:
            return fn(self, section, key, default)
        except:
            return default

    return wrapper_fn

class AnnParser(object):

    def __init__(self, filename):
        self.config = ConfigParser.RawConfigParser()
        self.config.read(filename)

        self.layer_sections = [s for s in self.config.sections() if s.startswith("Layer")]
        self.link_sections = [s for s in self.config.sections() if s.startswith("Link")]
        self.execution_order = [s for s in self.config.sections() if s == "Execution Order"]


    def extract_layers(self):
        self.layers = []

        for sec in self.layer_sections:
            self.layers.append(self._parse_layer(sec))

        return self.layers


    def extract_links(self):
        self.links = []

        for sec in self.link_sections:
            self.links.append(self._parse_link(sec))


    @fail
    def get_string(self, section, key, default):
        return self.config.get(section, key)

    @fail
    def get_int (self, section, key, default):
        return int(self.config.get(section, key))

    @fail
    def get_float (self, section, key, default):
        return float(self.config.get(section, key))

    @fail
    def get_func(self, section, key, default):
        str_repr = self.config.get(section, key)
        return funcs[str_repr]

    @fail
    def get_array(self, section, key, default):
        str_repr = self.config.get(section, key)
        return eval(str_repr)


    def get_layer(self, layers, section, key):
        layer_str = self.config.get(section, key)

        layer = self.find_layer(layers, layer_str)

        if not layer:
            raise ValueError("No valid layer selected in %s.%s" % (section, key))

        return layer


    def find_layer(self, layers, name):
        for layer in self.layers:
            if layer.name.lower() == name.lower():
                return layer

        return None

    def _parse_layer(self, section):
        # nodes = self.get_int(sec, "nodes")
        name = section.replace("Layer ", "")
        io_type = self.get_string(section, "io_type", None)
        nodes = self.get_int(section, "nodes", None)
        activation_function = self.get_func(section, "activation", Activation.sigmoid_tanh)
        return Layer(name, nodes, activation_function=activation_function, io_type=io_type)


    def _parse_link(self, section):
        pre = self.get_layer(layers, section, "pre")
        post = self.get_layer(layers, section, "post")
        topology = self.get_string(section, "topology", None)
        learning_rate = self.get_float(section, "learning_rate", 0.2)
        learning_rule = self.get_string(section, "learning_rule", None)
        arc_range = self.get_array(section, "arc_range", [-0.1, 0.1])
        weights = self.get_array(section, "weights", None)
        arcs = self.get_array(section, "arcs", None)
        return Link(pre, post, topology, arc_range, learning_rate, weights, arcs, learning_rule)

    def parse_execution_order(self):
        return self.get_array("Execution Order", "order", [])


if __name__ == "__main__":
    """
        Test for reading from the script file
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "script_ann", 
        metavar="ANNSCRIPT",
        type=str,
        nargs="*",
        default="scripts/static_ann.ini",
        help="Input ANN Script file")


    args = parser.parse_args()

    ini_parse = AnnParser(args.script_ann)

    layers = ini_parse.extract_layers()
    links = ini_parse.extract_links()

    gann = Ann(layers, links)
    gann.execution_order = ini_parse.parse_execution_order()

    def ra():
        return 1

    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
