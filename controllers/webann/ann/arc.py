

class Arc(object):

    def __init__(self,pre_node, post_node, current_weight = None, link = None):
        self.pre_node = pre_node
        self.post_node = post_node

        self.pre_node.outgoing.append(self)
        self.post_node.incomming.append(self)

        self.current_weight = current_weight
        self.init_weight = current_weight
        self.link = link

    def reset(self):
        """
            Return the weight to the initial weight.
        """
        self.current_weight = self.init_weight