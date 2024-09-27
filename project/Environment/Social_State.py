import networkx as nx
from typing import Mapping, Sequence
import copy

DEFAULT_ATTRIBUTE = 'link'

class SocialState:
    """The multi-DiGraph that represents the social state"""
    def __init__(self, 
                 name_list: Sequence[str], 
                 attr_dict: Mapping[str, Sequence[
                     tuple[str, str] | tuple[str, str, dict]
                 ]]) -> None:
        """
        Args:
            name_list: a list of names of nodes
            attr_list: a dict which the keys are the names of attributes,
                and the values are the edges of this attribute graph
        """

        self.name_list = list(name_list)
        self.G: dict[str, nx.DiGraph] = {}
        self.attr_list: list[str] = []

        for attr, edges in attr_dict.items():
            self.add_attr(attr, edges)
            
        self.record_init()
        
    def record_init(self):
        """record the initialization of the social state"""
        self.G_init = copy.deepcopy(self.G)
        self.name_list_init = self.name_list.copy()
        self.attr_list_init = self.attr_list.copy()

    def reset(self):
        """Reset the graphs of all attributes"""
        self.G = copy.deepcopy(self.G_init)
        self.name_list = self.name_list_init.copy()
        self.attr_list = self.attr_list_init.copy()
        

    def reset_attr(self, attr: str):
        """Take in the attribute name, and reset the graph of this attribute"""
        assert attr in self.attr_list, 'attribute not existed!'
        self.G[attr] = copy.deepcopy(self.G_init[attr])

    def clear(self):
        """Clear the graphs of all attributes"""
        for attr in self.attr_list:
            self.clear_attr(attr)

    def clear_attr(self, attr: str):
        """Take in the attribute name, and clear the graph of this attribute"""
        assert attr in self.attr_list, 'attribute not existed!'
        self.G[attr].remove_edges_from(list(self.G[attr].edges()))

    def add_attr(self, attr: str, edges: Sequence[tuple[str, str] | tuple[str, str, dict]]):
        """Add a new attribute, build a new graph for this attribute"""
        assert attr not in self.attr_list, 'attribute already existed!'
        self.G[attr] = nx.DiGraph(attribute = attr)
        self.attr_list.append(attr)
        self.G[attr].add_nodes_from(
            [(name, {'creator': name}) for name in self.name_list]
        )
        self.G[attr].add_edges_from(edges)

    # adding a new node
    def add_v(self, 
              name: str, 
              creator: str | None = None, 
              node_attribute: dict[str, dict[str, object]] = {DEFAULT_ATTRIBUTE: {}}):
        """Add a new node for all attribute graphs

        Args:
            name: the name of the node
            creator: the creator of the node, should be another existed node or None
            node_attribute: attribute of the new node
        """
        
        assert name not in self.name_list, 'name already exists!'
        assert (creator is None) or (creator in self.name_list), 'creator not exists!'
        self.name_list.append(name)

        for graph_attr, node_attr in node_attribute.items():
            assert 'creator' not in node_attr.keys(), '``creator`` is a built-in node attribute!'
            node_attr['creator'] = creator or name
            self.G[graph_attr].add_node(name, **node_attr)

    def remove_v(self, name: str):
        """Remove a node named ``name`` in all attribute graphs"""
        assert name in self.name_list, 'name not exists!'
        self.name_list.remove(name)

        for attr in self.attr_list:
            self.G[attr].remove_node(name)

    def set(self, graph_attr_list: Sequence[str], i: str, j: str, value=None):
        """Add an edge weighted ``value`` between 
        node ``i`` and node ``j`` in the ``attr`` graph"""
        for attr in graph_attr_list:
            if value is None:
                self.G[attr].add_edge(i, j)
            else:
                self.G[attr].add_edge(i, j, weight=value)

    def remove(self, attr: str, i: str, j: str):
        """Remove an edge between node ``i`` and node ``j`` in the ``attr`` graph"""
        self.G[attr].remove_edge(i, j)

    def adj_matrix(self, attr: str = DEFAULT_ATTRIBUTE, weight_key: str | None = "weight"):
        """Return the adjacency matrix of the ``weight_key`` edges of ``attr`` matrix"""
        return nx.adjacency_matrix(self.G[attr], weight=weight_key).toarray()

    def obs_process(self, obs: dict):
        """Implement the influence of the social state to obseravtion"""
        return obs
    
    def reward_process(self, reward: dict[str, float]):
        """Implement the influence of the social state to reward"""
        return reward
