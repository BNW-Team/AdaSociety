import numpy as np
from typing import Sequence, Mapping
from .utils import mapping2array, map_index

NAME = "name"
INIT_POS = "init_pos"
EVENT_IO = "event_io"
INPUT = "inputs"
OUTPUT = "outputs"
REQUIREMENT = "requirements"
REQUIREMENT_ARRAY = "requirement_array"
AVAIL_INTERVAL = "avail_interval"

class EventBase:
    """The base class for Events"""
    inputs: dict[str, int] = {}
    outputs: dict[str, int] = {}
    requirements: dict[str, int] = {}
    
    def __init__(self, args: Mapping) -> None:
        """
        Args:
            name: identify resource types.
            init_pos: initial positions. if None, random generate.
            Event_IO: Input and Output for this event.
        """

        self.name :str = args.get(NAME, "base")
        self.init_pos : Sequence[int, int] | None = args.get(INIT_POS, None)
        self.Event_IO : Sequence[int] = args[EVENT_IO]
        self.requirement_array: Sequence[int] = args[REQUIREMENT_ARRAY]
        self.avail_interval: int = args.get(AVAIL_INTERVAL, 1)

class Woodwork(EventBase):
    inputs = {"wood": 1, "stone": 1}
    outputs = {"axe": 1}
    requirements = {}
    
    def __init__(self, args) -> None:
        super().__init__(args)
        self.name = "woodwork"

class HammerCraft(EventBase):
    inputs = {"wood": 1, "stone": 1}
    outputs = {"hammer" : 1}
    requirements = {}
    
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "hammercraft"

class TorchCraft(EventBase):
    inputs = {"wood": 1, "coal": 1}
    outputs = {"torch": 1}
    requirements = {"coal": 1}
    
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "torchcraft"

class SteelMaking(EventBase):
    inputs = {"iron": 1, "coal": 1}
    outputs = {"steel": 1}
    requirements = {"iron": 1}
    
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "steelmaking"

class Potting(EventBase):
    inputs = {"clay": 2, "coal": 1}
    outputs = {"pottery": 1}
    requirements = {"clay": 1}
    
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "potting"

class ShovelCraft(EventBase):
    inputs = {"steel": 2, "wood": 2}
    outputs = {"shovel": 1}
    requirements = {"steel": 1}
    
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "shovelcraft"
        
class PickaxeCraft(EventBase):
    inputs = {"steel": 3, "wood": 2}
    outputs = {"pickaxe": 1}
    requirements = {"steel": 1}
    
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "pickaxecraft"

class CutterCraft(EventBase):
    inputs = {"steel": 2, "stone": 3}
    outputs = {"cutter": 1}
    requirements = {"steel": 1}
    
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "cuttercraft"
        
class GemCutting(EventBase):
    inputs = {"cutter": 1, "gem_mine": 1}
    outputs = {"gem": 1}
    requirements = {"cutter": 1, "gem_mine": 1}
    
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "gemcutting"
        
class TotemMaking(EventBase):
    inputs = {"gem": 2, "pottery": 1, "steel": 1}
    outputs = {"totem": 2}
    requirements = {"gem": 1}
    
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "totemmaking"

NAME2EVENT: dict[str, EventBase] = {
    "woodwork": Woodwork,
    "hammercraft": HammerCraft,
    "torchcraft": TorchCraft,
    "steelmaking": SteelMaking,
    "potting": Potting,
    "shovelcraft": ShovelCraft,
    "pickaxecraft": PickaxeCraft,
    "cuttercraft": CutterCraft,
    "gemcutting": GemCutting,
    "totemmaking": TotemMaking
}

def event_grounding(args_list, resource_name_list: list[str]) -> list[EventBase]:
    """Take in the list of event feature dictionaries, and returns an Event list
    
    Args:
        args_list: the list of the feature of each event
        resource_name_list:  a list that shows the resources contained in this game
    
    Returns:
        events: a list of Events which are created according to those features
        name_list: a list of name corresponding to the Event in ``events``

    Examples:
        args_list = [
            {
                "name": "woodwork",
                "init_pos": None,
                "inputs": {"wood": 1, "stone": 1},
                "outputs": {"axe": 1}
            },
            {
                "name": "firework",
                "init_pos": None,
                "inputs": {"wood": 2, "dynamite": 1},
                "outputs": {"firework": 1}
            }
        ]
    """

    events = []
    for args in args_list:
        name = args[NAME]
        
        if name in NAME2EVENT.keys():
            # print(f"Choose built-in Event: {name} - inputs: {NAME2EVENT[name].inputs}, outputs: {NAME2EVENT[name].outputs}")
            args[EVENT_IO], args[REQUIREMENT_ARRAY] = parse_event(
                resource_name_list, 
                NAME2EVENT[name].inputs, 
                NAME2EVENT[name].outputs, 
                NAME2EVENT[name].requirements, 
                name
            )
            events.append(NAME2EVENT[name](args))
        else:
            args[EVENT_IO], args[REQUIREMENT_ARRAY] = parse_event(
                resource_name_list, 
                args[INPUT], 
                args[OUTPUT], 
                args[REQUIREMENT], 
                name
            )
            events.append(EventBase(args))
            
    return events

def parse_event(resource_name_list: list[str], 
                input: Mapping[str, int], 
                output: Mapping[str, int], 
                requirement: Mapping[str, int], 
                name: str):
    '''Parsing input and output parameters of an Event and construct the event_io arrays
    
    Args:
        resource_name_list:  a list that shows the resources contained in this game
        input: input resources (name and number)
        output: output resources (name and number)
        name: the name of the checking Event

    Returns:
        The event_io arrays of the checking Event
    '''

    resource2id = map_index(resource_name_list)
    event_io = np.zeros((len(resource_name_list), ))
    for resource, num in input.items():
        assert resource in resource_name_list, f'input {resource} of Event {name} not in the resource list!'
        event_io[resource2id[resource]] -= num
    for resource, num in output.items():
        assert resource in resource_name_list, f'output {resource} of Event {name} not in the resource list!'
        event_io[resource2id[resource]] += num
        
    requirement_array = mapping2array(requirement, resource2id, 0, "error")
    return event_io, requirement_array