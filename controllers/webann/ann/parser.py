import ConfigParser
from functools import wraps, partial
from layer import *
from link import *
from ann import Ann
from ann_modules import *


funcs = {
    'log': Activation.sigmoid_log,
    'tanh': Activation.sigmoid_tanh,
    'step': Activation.step,
    'step_0': partial(Activation.step, T=0),
    'step_5': partial(Activation.step, T=0.5),
    'step_8': partial(Activation.step, T=0.8),
    'linear': Activation.linear,
    'pos_linear': Activation.pos_linear
}

rules = {
    'hebbian': LearningRule.hebbian,
    'general_hebb': LearningRule.general_hebb,
    'oja': LearningRule.oja
}

def fail(fn):

    @wraps(fn)
    def wrapper_fn(self, section, key, default):
        try:
            tmp = fn(self, section, key, default)
            if tmp == "None" or not(tmp):
                return None
            return tmp
        except:
            return default

    return wrapper_fn

class AnnParser(object):

    def __init__(self, filename):
        self.config = ConfigParser.RawConfigParser()
        self.config.read(filename)

        self.layer_sections = [s for s in self.config.sections() if s.startswith("Layer")]
        self.link_sections = [s for s in self.config.sections() if s.startswith("Link")]
        self.inhibitory_sections = [s for s in self.config.sections() if s.startswith("Inhibitory")]
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

    def extract_modules(self):
        for sec in self.inhibitory_sections:
            self.layers.append(self._parse_inhibitory(sec))


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

        if str_repr == "step":
            step_v = self.get_float(section, "step", 0.5) 
            step_v = step_v if step_v else 0

            return partial(Activation.step, T=step_v)


        return funcs[str_repr]

    @fail
    def get_rule(self, section, key, default):
        str_repr = self.config.get(section, key)
        return rules[str_repr]

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

    def _parse_inhibitory(self, section):
        name = section.replace("Inhibitory ", "")
        neg = self.get_float(section, "neg", -4)
        pos = self.get_float(section, "pos", 0.1)

        up = self.get_layer(self.layers, section, "up")
        down = self.get_layer(self.layers, section, "down")

        activation_function = self.get_func(section, "activation", Activation.step)

        return Inhibitory(name, activation_function=activation_function, neg=neg, pos=pos, up=Link(pre_layer=up), down=Link(post_layer=down))


    def _parse_link(self, section):
        pre = self.get_layer(self.layers, section, "pre")
        post = self.get_layer(self.layers, section, "post")
        topology = self.get_string(section, "topology", None)
        learning_rate = self.get_float(section, "learning_rate", 0.2)
        learning_rule = self.get_rule(section, "learning_rule", LearningRule.general_hebb)
        arc_range = self.get_array(section, "arc_range", [-0.1, 0.1])
        weights = self.get_array(section, "weights", None)
        arcs = self.get_array(section, "arcs", None)

        return Link(pre, post, topology, arc_range, learning_rate, weights, arcs, learning_rule)

    def parse_execution_order(self):
        return self.get_array("Execution Order", "order", [])

    def create_ann(self):
        self.extract_layers()

        self.extract_modules()

        self.extract_links()

        return Ann(self.layers, self.links, self.parse_execution_order())

    @staticmethod
    def reverse_func_lookup(fn):

        for key, func in funcs.items():
            if func == fn:
                return key

        return None

    @staticmethod
    def reverse_rule_lookup(fn):

        for key, func in rules.items():
            if func == fn:
                return key

        return None

    @staticmethod
    def insert_link(cfg, link, i, use_updated_values = False):
        section = 'Link %s' % i
        cfg.add_section(section)

        cfg.set(section, 'arc_range', str(link.arc_range))

        if link.learning_rule:
            cfg.set(section, 'learning_rule', reverse_rule_lookup(link.learning_rule))

        if link.learning_rate:
            cfg.set(section, 'learning_rate', link.learning_rate)
        
        if link.topology:
            cfg.set(section, 'topology', link.topology)

        
        if use_updated_values:
            cfg.set(section, 'weights', link.export_weights())

        elif link.init_weights or use_updated_values:
            cfg.set(section, 'weights', link.init_weights)

        if use_updated_values:
            cfg.set(section, 'arcs', link.export_arcs())

        elif link.init_arcs:
            cfg.set(section, 'arcs', link.init_arcs)

        cfg.set(section, 'post', link.post_layer.name)
        cfg.set(section, 'pre', link.pre_layer.name)

    @staticmethod
    def insert_execution_order(cfg, execution_order):
        section = 'Execution Order'
        cfg.add_section(section)
        cfg.set(section, 'order', str([i.name for i in execution_order]))

    @staticmethod
    def insert_layer(cfg, layer):
        section = 'Layer %s' % layer.name
        cfg.add_section(section)

        if hasattr(layer.activation_function, 'func'):
            # Is a partial
            cfg.set(section, 'activation', AnnParser.reverse_func_lookup(layer.activation_function.func))
            cfg.set(section, 'step', float(layer.activation_function.keywords['T']))

        else:
            cfg.set(section, 'activation', AnnParser.reverse_func_lookup(layer.activation_function))

        cfg.set(section, 'nodes', len(layer.nodes))

        if layer.type: 
            cfg.set(section, 'io_type', layer.type)

    @staticmethod
    def insert_inhibitory(cfg, module):

        section = 'Inhibitory %s' % module.name
        cfg.add_section(section)

        if hasattr(module.activation_function, 'func'):
            # Is a partial
            cfg.set(section, 'activation', AnnParser.reverse_func_lookup(module.activation_function.func))
            cfg.set(section, 'step', float(module.activation_function.keywords['T']))

        else:
            cfg.set(section, 'activation', AnnParser.reverse_func_lookup(module.activation_function))

        cfg.set(section, "neg", module.neg)
        cfg.set(section, "pos", module.pos)
        cfg.set(section, "up", module.up.name)
        cfg.set(section, "down", module.down.name)


    @staticmethod
    def export(ann, filename, use_updated_values = False):
        config = ConfigParser.RawConfigParser()

        for layer in ann.layers:
            if hasattr(layer, "links"):
                AnnParser.insert_inhibitory(config, layer)
            else:
                AnnParser.insert_layer(config, layer)

        for i, link in enumerate(ann.links):
            AnnParser.insert_link(config, link, i, use_updated_values)
        
        AnnParser.insert_execution_order(config, ann.execution_order)

        # Writing our configuration file to 'filename'
        with open(filename, 'wb') as configfile:
            config.write(configfile)


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

    # ini_parse = AnnParser(args.script_ann)
    # gann = ini_parse.create_ann()

    # AnnParser.export(gann, "scripts/test.ini")

    ini_parse = AnnParser("scripts/test.ini")
    gann = ini_parse.create_ann()

    import random
    def ra():
        return random.random()

    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
    print gann.recall([ra() for i in range(30)])
