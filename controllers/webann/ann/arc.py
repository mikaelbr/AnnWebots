

class Arc(object):

    def __init__(self,pre_node,post_node,current_weight,init_weight,link):
        self.pre_node = pre_node
        self.post_node = post_node
        self.current_weight = current_weight
        self.init_weight = init_weight
        self.link = link
