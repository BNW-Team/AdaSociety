# this file defines the Action class and it's Derived class.
# The Action class is the father class. Every Action should be derived from the Action class.
import numpy as np
import copy
from typing import Sequence

class Action:
    """The base class for actions"""
    pass


class MovementAction(Action):
    """The movement action"""
    def __init__(self, dx: int, dy: int):
        """
        Args:
            dx: movement in x-axis
            dy: movement in y-axis
        """

        super().__init__()
        self.dx: int = dx
        self.dy: int = dy


class Pick_Dump_Action(Action):
    """The pick-or-dump action"""
    def __init__(self, resources_vec):
        """
        Args:
            resources_vec: a vector of resource quantities that agents
            want to put in their inventory, where the negative number
            means taking the resource out of their inventory
        """

        super().__init__()
        self.resources_vec = np.array(copy.deepcopy(resources_vec))
        
class Produce_Action(Action):
    """The produce action on an Event"""
    def __init__(self) -> None:
        """
        Execute this action on an Event will execute the Event IO process
            in your inventory
        """
        super().__init__()


class Communication_Action(Action):
    """The communication action"""
    def __init__(self, towards: Sequence[str], comm_vec: np.array):
        """
        Args:
            towards: A list of players' names that agents want to communicate
            comm_vec: the content of the communication
        """

        super().__init__()
        self.comm_vec = copy.deepcopy(comm_vec)
        self.towards = towards


class SocialConnect_Action(Action):
    """Connect and disconnect in social state"""
    def __init__(self, 
                 attribute: str, 
                 source: str,
                 connect: str | None, 
                 disconnect: str | None, 
                 weights: float | None = None) -> None:
        """
        Args:
            attribute: action will operate in this graph
            connect: the name of the node that the agent want to connect
            disconnect: the name of the node that the agent want to disconnect
            weights: the weight of the edges when connecting node
        """

        super().__init__()
        self.attribute = attribute
        self.source = source
        self.connect =  connect
        self.disconnect = disconnect
        self.weights = weights


class CreateSocialNode_Action(Action):
    """Create a node in social state"""
    def __init__(self, 
                 name: str, 
                 node_attribute:dict[str, dict[str, object]]) -> None:
        """
        Args:
            name: the name of the creating node
            node_attribute: key - graph attribute
                value - attribute of the new node in graph ``key``
        """

        super().__init__()
        self.name = name
        self.node_attribute = node_attribute
        

class RemoveSocialNode_Action(Action):
    """Remove a node in social state"""
    def __init__(self, name: str) -> None:
        """
        Args:
            name: the name of the removing node
        """

        super().__init__()
        self.name = name