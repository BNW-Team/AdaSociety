import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Mapping
from .utils import mapping2array, map_index

NAME = "name"
INIT_POS_LIST = "init_pos_list"
INIT_NUM_LIST = "init_num_list"
REQUIREMENT = "requirements"
REQUIREMENT_ARRAY = "requirement_array"

class ResourceBase:
    """The base class for Resources"""
    requirements: dict[str, int] = {}
    def __init__(self, args: Mapping) -> None:
        """
        Attributes:
            name: identify resource types.
            init_pos_list: initial positions. if None, random generate.
            init_num_list: initial number of resources in each initial positions.
        """

        self.name : str = args.get(NAME, "base")
        self.init_pos_list : Sequence[Sequence[int, int]] | None = args.get(INIT_POS_LIST, None)
        self.init_num_list : Sequence[int] = args[INIT_NUM_LIST]
        self.requirement_array: Sequence[int] = args[REQUIREMENT_ARRAY]


class Wood(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "wood"

class Stone(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "stone"

class Axe(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "axe"

class Hammer(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "hammer"
        
class Coal(ResourceBase):
    requirements = {"hammer": 1}
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "coal"
        
class Torch(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "torch"
        
class Iron(ResourceBase):
    requirements = {"torch": 1}
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "iron"

class Steel(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "steel"
        
class Shovel(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "shovel"
        
class Pickaxe(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "pickaxe"
        
class GemMine(ResourceBase):
    requirements = {"torch": 1, "pickaxe": 1}
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "gem_mine"

class Clay(ResourceBase):
    requirements = {"shovel": 1}
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "clay"
        
class Pottery(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "pottery"

class Cutter(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "cutter"

class Gem(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "gem"
        
class Totem(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "totem"
    
class Power(ResourceBase):
    def __init__(self, args: Mapping) -> None:
        super().__init__(args)
        self.name = "power"     

NAME2RESOURCE: dict[str, ResourceBase] = {
    "wood": Wood,
    "stone": Stone,
    "axe": Axe,
    "power": Power,
    "hammer": Hammer,
    "coal": Coal,
    "torch": Torch,
    "iron": Iron,
    "steel": Steel,
    "shovel": Shovel,
    "pickaxe": Pickaxe,
    "gem_mine": GemMine,
    "clay": Clay,
    "pottery": Pottery,
    "cutter": Cutter,
    "gem": Gem,
    "totem": Totem
}

def resource_grounding(args_list: Sequence[Mapping]) -> tuple[list[ResourceBase], list[str], NDArray]:
    """Take in the list of resource feature dictionaries, and returns an Resource list
    
    Args:
        args_list: the list of the feature of each resource
    
    Returns:
        resources: a list of Resources which are created according to those features
        name_list: a list of name corresponding to the Resource in ``resources``

    Examples:
        see ``args/basic_args.py``
    """
    
    resources = []
    name_list = []
    all_requirement = []
    for args in args_list:
        name = args[NAME]
        assert name not in name_list, 'Duplicate resource name!'
        name_list.append(name)

    for args in args_list:
        name = args[NAME]
        if name not in NAME2RESOURCE:
            requirement = args.get(REQUIREMENT, {})
            args[REQUIREMENT_ARRAY] = mapping2array(requirement, map_index(name_list))
            resources.append(NAME2RESOURCE[name](args))
        else:
            requirement = NAME2RESOURCE[name].requirements
            args[REQUIREMENT_ARRAY] = mapping2array(requirement, map_index(name_list))
            resources.append(ResourceBase(args))
        all_requirement.append(args[REQUIREMENT_ARRAY].copy())
            
    return resources, name_list, np.array(all_requirement)