import numpy as np
from typing import Sequence
from .utils import map_index, mapping2array

NAME = "name"
ALIVE = "alive"
INIT_POS = "init_pos"
OBS_RANGE = "obs_range"
RESOURCE_VALUE = "resource_value"
INVENTORY_TOTAL_MAX = "inventory_total_max"
INITIAL_INVENTORY = "initial_inventory"
INVENTORY_OBJECT_MAX = "inventory_object_max"
ARRAY = "_array"

from .Map import MAX_ITEM_NUM



class Player:
    def __init__(self, args: dict) -> None:
        """All attribute for players

        Args:
            args: a dictionary that contains the attribute of players  
            
        Attributes:
            name: the player's name
            init_pos: a 2-tuple of initial position. If None, random generated position
            obs_range: a 2-tuple of observation range in x-axis and y-axis
            resource_value: the value of each resources to this player 
            inventory_object_max: the maximum number of resources that a player can possess
            inventory_total_max: the total maximum number of resources that a player can possess
            alive (bool): alive status
        """

        self.name:str = args[NAME]
        self.init_pos = args.get(INIT_POS, None)
        self.obs_range = args[OBS_RANGE]
        self.resource_value = args[RESOURCE_VALUE+ARRAY]
        self.inventory_max = args[INVENTORY_OBJECT_MAX+ARRAY]
        self.inventory_total_max = args.get(INVENTORY_TOTAL_MAX, MAX_ITEM_NUM)
        self.initial_inventory = args[INITIAL_INVENTORY+ARRAY]
        self.inventory = self.initial_inventory.copy()
        self.alive = args[ALIVE]

    @property
    def credit(self):
        """Return a player's credit"""
        return np.sum(self.resource_value*self.inventory)
    
    def reset_inventory(self):
        """Reset the player's inventory to all zero"""
        self.inventory = self.initial_inventory.copy()


class PlayerPool:
    def __init__(self, player_features: Sequence[dict], resource_name_list: Sequence[str]) -> None:
        """A class control all players in this game.
        
        Args:
            player_features: a list where each elements are a dictionary contains
                the player features
            resource_name_list: a list that shows the resources contained in this game

        Attributes:
            player_features: a list where each elements are a dictionary contains
                the player features
            resource_name_list: a list that shows the resources contained in this game
            player_pool: a list that contains all players. Every player's index in this 
                list is considered as the player's id. 
            player_count: the number of players
            alive_player_id: all alive players' ids.
            player_id2name: A dictionary that takes players' ids as keys, and players' names
                as values
            player_name2id: A dictionary that takes players' names as keys, and players' ids
                as values

        Examples:
            player_features = [
                {
                    "init_pos": None,
                    "obs_range": (1, 1),
                    "alive": True,
                    "resource_value": {
                         "wood": 1,
                         "stone": 1, 
                         "axe": 10,
                    },
               },
               {
                    "init_pos": [5,5],
                    "obs_range": (2, 3),
                    "alive": True,
                    "resource_value": {
                         "wood": -1,
                         "stone": 10, 
                         "axe": 1,
                    },
               }
            ]
        """

        self.player_features = list(player_features)
        self.resource_name_list = resource_name_list
        self.player_pool = self.player_grounding(player_features)
        self.alive_player_id = [i for i, player in enumerate(self.player_pool) if player.alive]
        self.player_id2name = {i: player.name for i, player in enumerate(self.player_pool)}
        assert len(list(self.player_id2name.values())) \
            == len(set(self.player_id2name.values())), 'Duplicate name!'
        self.player_name2id = {player.name: i for i, player in enumerate(self.player_pool)}

    @property
    def player_count(self):
        return len(self.player_pool)
    
    def add_player(self, player_feature: dict):
        """Add a player into the player pool

        Args:
            player_feature: the feature dictionary of the player
        """

        self.player_features.append(player_feature)
        player = self.player_grounding([player_feature])[0]
        assert player.name not in self.player_name2id.keys(), 'Duplicate name!'
        self.player_pool.append(player)
        id = self.player_count - 1
        if self.player_pool[-1].alive:
            self.alive_player_id.append(id)
        self.player_id2name[id] = player.name
        self.player_name2id[player.name] = id

    def player_grounding(self, args_list: Sequence[dict]) -> list[Player]:
        """Take in the list of feature dictionaries, and returns a Player list
        
        Args:
            args_list: the list of player's feature
        
        Return:
            a list of Players which are created according to those features
        """

        resource2id = map_index(self.resource_name_list)
        player_list = []
        for args in args_list:
            args[INITIAL_INVENTORY+ARRAY] = mapping2array(args[INITIAL_INVENTORY],resource2id, 0)
            args[RESOURCE_VALUE+ARRAY] = mapping2array(args[RESOURCE_VALUE], resource2id, 0)
            args[INVENTORY_OBJECT_MAX+ARRAY] = mapping2array(args[INVENTORY_OBJECT_MAX], resource2id, MAX_ITEM_NUM)
            player_list.append(Player(args))
        return player_list
        