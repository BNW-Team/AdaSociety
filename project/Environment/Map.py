# Core file. All the map related class should be implemented here.

import numpy as np
import pygame
import random
from typing import Sequence, Mapping, Any
import itertools

MAX_ITEM_NUM = 32767

class Index:
    def __init__(self, Resource_num) -> None:
        self.Agent_ID = (0, 1)
        self.Event_ID = (1, 2)
        self.Block = (2, 3)
        self.Event_avail_interval = (3, 4)
        self.Event_avail_countdown = (4, 5)
        self.Event_IO = (5, 5 + Resource_num)
        self.Event_requirement = (5 + Resource_num, 5 + Resource_num*2)
        self.Max_Res = (5 + Resource_num*2, 5 + Resource_num*3)
        self.Cur_Res = (5 + Resource_num*3, 5 + Resource_num*4)
        self.Layer_num = 5 + Resource_num*4
        
from .Game_Action import (
    Action, MovementAction, Pick_Dump_Action, Produce_Action, Communication_Action, 
    SocialConnect_Action, CreateSocialNode_Action, RemoveSocialNode_Action
    )
from .Player import PlayerPool, Player
from .Event import event_grounding
from .Resource import resource_grounding
from .Communication import CommunicationBoard
from .Social_State import SocialState, DEFAULT_ATTRIBUTE
from .Stage import StageController
from .utils import mapping2array, map_index

# For infinite generated map. Now pass. A Big_Map contains a 2D array of Chunks.
# TODO: implement infinite generated map.
        
class Big_Map:
    pass

# Definition for a single chunk

class Chunk:
    """The main logic of the environment"""

    def __init__(
            self, 
            size: tuple[int, int] = (20, 20),
            Block_num: int = 0,
            Block_pos: Sequence[tuple[int, int]] = [],
            Resource_feature: Sequence[Mapping[str, object]] = [], 
            Player_feature: Sequence[Mapping[str, object]] = [], 
            Event_feature: Sequence[Mapping[str, object]] = [],
            grid_object_max: Mapping[str, int] = [],
            communication_length: int = 8,
            Event_random_output_range: int = 1,
            Social_node: Sequence[str] = [],
            Social_attr: Mapping[str, Sequence[
                tuple[str, str] | tuple[str, str, dict]
            ]] = {DEFAULT_ATTRIBUTE: []},
            With_Visual: bool = True,
            Abstract_mapping=[],
            around_chunk=[],
            ):
        """Initialize a chunk

            Args:
                size: the size of map
                Block_num: the number of blocks
                Block_pos: the position of all blocks
                Resource_feature: a list of resource configs
                Player_feature: a list of player configs
                Event_feature: a list of event configs
                grid_object_max: a mapping showing the maximum number of resources that a grid
                    can contain, where the key is the name of resources.
                communication_length: the length of the communication channel
                Event_random_output_range: the range that the output of the events will locate
                    [WARNING: Useless now. The implementation now is equivalent to 
                    Event_random_output_range = 0 always]
                    (Note: May put in the event's configs?)
                Social_node: A list of node name for social state.
                Social_attr: A dict of {attribute name: list of edges} for social state. 
                    One graph is generated for each attribute
                Abstract_mapping: inputs the Abstract_mapping dict defined by certain rules
                    [WARNING: may be buggy now]
                around_chunk: a future input for infinite map generation, Which the current chunk
                    defines by the chunk around it
                    [WARNING: Useless now]

            Examples:
                see ``args`` directory
        """

        
        self.size: Sequence[int, int] = size
        self.block_num = Block_num
        self.block_pos = list(Block_pos)
        self.Resource, self.resource_name_list, self.resource_requirement_array = resource_grounding(Resource_feature)
        self.Resource_num = len(self.Resource)
        self.Event = event_grounding(Event_feature, self.resource_name_list)
        self.Event_num = len(self.Event)
        self.Player_pool = PlayerPool(Player_feature, self.resource_name_list)
        self.Communication = CommunicationBoard(len(self.Player), communication_length, self.Player_pool.player_name2id)
        self.index = Index(self.Resource_num)
        self.social_state = SocialState(Social_node, Social_attr)
        self.stage_controller = StageController(self.index, self.resource_requirement_array)
        
        self.grid_object_max = grid_object_max
        self.init_chunk()
        
        self.tilechunk = np.zeros((self.size[0], self.size[1]), dtype=int)
        self.agents_place = -np.ones((len(self.Player), 2), dtype=np.int16)
        self.events_place = -np.ones((self.Event_num, 2), dtype=np.int16)

        # may be buggy
        self.Abstract_mapping = Abstract_mapping
        self.around_chunk = around_chunk
        self.With_Visual=With_Visual
        self.Event_random_output_range=Event_random_output_range

        self.background_tilechunk = np.zeros((self.size[0], self.size[1]), dtype=int)
        self.episodes = 0
    
    @property
    def Player(self):
        return self.Player_pool.player_pool
    
    @property
    def player_name(self):
        return self.Player_pool.player_id2name
    
    @property
    def agent_layer(self):
        return self.chunk[:, :, self.index.Agent_ID[0] : self.index.Agent_ID[1]]

    @property
    def event_layer(self):
        return self.chunk[:, :, self.index.Event_ID[0] : self.index.Event_ID[1]]
    
    @property
    def block_layer(self):
        return self.chunk[:, :, self.index.Block[0] : self.index.Block[1]]
    
    @property
    def event_io_layer(self):
        """input is negative, while output is positive"""
        return self.chunk[:, :, self.index.Event_IO[0] : self.index.Event_IO[1]]
    
    @property
    def event_requirement_layer(self):
        return self.chunk[:, :, self.index.Event_requirement[0] : self.index.Event_requirement[1]]
    
    @property
    def event_avail_interval_layer(self):
        return self.chunk[:, :, self.index.Event_avail_interval[0] : self.index.Event_avail_interval[1]]

    @property
    def event_avail_countdown_layer(self):
        return self.chunk[:, :, self.index.Event_avail_countdown[0] : self.index.Event_avail_countdown[1]]
    
    @property
    def max_resource_layer(self):
        return self.chunk[:, :, self.index.Max_Res[0] : self.index.Max_Res[1]]
    
    @property
    def current_resource_layer(self):
        return self.chunk[:, :, self.index.Cur_Res[0] : self.index.Cur_Res[1]]
    
    @property
    def default_social_graph(self):
        return self.social_state.G[DEFAULT_ATTRIBUTE]
    
    def agent_pos(self, player_id: int):
        return self.agents_place[player_id]
    
    def exist_player(self, pos):
        return self.agent_layer[pos[0], pos[1]] > 0 
    
    def exist_event(self, pos):
        return self.event_layer[pos[0], pos[1]] > 0
    
    def exist_resource(self, pos):
        return (self.current_resource_layer[pos[0], pos[1]] > 0).any()
    
    def exist_resource_id(self, pos, resource_id):
        return self.current_resource_layer[pos[0], pos[1], resource_id] > 0
    
    def possess_resource(self, player_name, resource_id):
        return self.Player[self.Player_pool.player_name2id[player_name]].inventory[resource_id] > 0
    
    def load_event_countdown(self, pos):
        self.event_avail_countdown_layer[pos[0], pos[1]] = self.event_avail_interval_layer[pos[0], pos[1]]

    def init_chunk(self):
        '''Initialize a new chunk'''
        self.chunk = np.zeros((self.size[0], self.size[1], self.index.Layer_num), dtype=np.int32)
        self.max_resource_layer[:,:] = mapping2array(self.grid_object_max, 
                                                     map_index(self.resource_name_list), 
                                                     MAX_ITEM_NUM)

    def init_block(self) -> list[tuple[int, int]]:
        """Initialize blocks
        
        Random choose ``self.block_num`` blocks from ``self.block_pos``.

        If ``len(self.block_pos)`` > ``self.block_num``,
            add all blocks in ``self.block_pos``.

        Return:
            A list of coordinates without blocks
        """

        blank_pos = set(itertools.product(range(0, self.size[0]), range(0, self.size[1])))
        if self.block_num < 0:
            raise ValueError("block_num must >= 0")
        elif self.block_num == 0 or len(self.block_pos) == 0:
            return list(blank_pos)
        elif self.block_num < len(self.block_pos):
            block_pos = random.sample(self.block_pos, self.block_num)
        else:
            block_pos = self.block_pos
        for pos in block_pos:
            blank_pos.remove(pos)
            
        subscript = np.array(block_pos).T
        self.block_layer[subscript[0], subscript[1]] = 1
        
        return list(blank_pos)
    
    def init_extra_block(self):
        """Randomly add extra blocks
        
        If ``len(self.block_pos)`` > ``self.block_num``,
            add extra ``self.block_num - len(self.block_pos)`` blocks randomly
        """
        if self.block_num > len(self.block_pos):
            block_pos = random.sample(self.blank_pos, self.block_num - len(self.block_pos))
            for pos in block_pos:
                self.blank_pos.remove(pos)

            subscript = np.array(block_pos).T
            self.block_layer[subscript[0], subscript[1]] = 1
            
    def init_player_pos(self):
        '''Initialize player positions
        
        If ``event.init_pos`` is None, player's position is randomly generated 
        and directly painted on ``self.agent_layer``
        '''

        pos_pool = random.sample(self.blank_pos, len(self.Player_pool.alive_player_id))

        random_pos_player_id = []
        for i in self.Player_pool.alive_player_id:
            player = self.Player[i]
            if player.init_pos is None:
                random_pos_player_id.append(i)
            else:
                assert self.exist_player(player.init_pos), "Multiple players in one cell!"
                assert self.block_layer[player.init_pos[0], player.init_pos[1]] == 0, \
                f"Player and block in the same cell ({player.init_pos[0]}, {player.init_pos[1]})"
                self.load_agent_pos(i, player.init_pos)
        
        count = 0
        for i in random_pos_player_id:
            while self.exist_player(pos_pool[count]):
                count += 1
            self.load_agent_pos(i, pos_pool[count])
        
    def load_agent_pos(self, index, pos):
        self.agents_place[index] = pos
        self.agent_layer[pos[0], pos[1]] = index + 1

    def init_event_pos(self):
        '''Initialize event positions
        
        If ``event.init_pos`` is None, event's position is randomly generated 
        and directly painted on ``self.event_layer`` & ``self.event_io_layer``
        '''

        pos_pool = random.sample(self.blank_pos, len(self.Event))

        random_pos_event_id = []
        for i, event in enumerate(self.Event):
            if event.init_pos is None:
                random_pos_event_id.append(i)
            else:
                assert self.exist_event(event.init_pos), \
                f"Multiple events in one cell ({event.init_pos[0]}, {event.init_pos[1]})!"
                assert self.block_layer[event.init_pos[0], event.init_pos[1]] == 0, \
                f"Event and block in the same cell ({event.init_pos[0]}, {event.init_pos[1]})"
                self.load_event_pos(i, event.init_pos)     
        
        count = 0
        for i in random_pos_event_id:
            while self.exist_event(pos_pool[count]): 
                count += 1
            self.load_event_pos(i, pos_pool[count])
            
    def load_event_pos(self, index, pos):
        """Load an event to a grid of the map"""
        self.events_place[index] = pos
        self.event_layer[pos[0], pos[1]] = index + 1
        self.event_avail_interval_layer[pos[0], pos[1]] = self.Event[index].avail_interval
        self.event_io_layer[pos[0], pos[1]] = self.Event[index].Event_IO
        self.event_requirement_layer[pos[0], pos[1]] = self.Event[index].requirement_array
        self.blank_pos.remove((pos[0], pos[1]))

    def init_resource_pos(self):
        '''Initialize resource positions
        
        Different number of resources are put into different resource bases according to ``resource.init_num_list``
        '''

        random_pos_resource_id: list[int] = []
        for i, resource in enumerate(self.Resource):
            if resource.init_pos_list is None:
                random_pos_resource_id.append(i)
            else:
                assert len(resource.init_pos_list) == len(resource.init_num_list), \
                    '"init_pos_list" should be of the same length of "init_num_list" when "init_pos_list" is not None'
                for pos, num in zip(resource.init_pos_list, resource.init_num_list):
                    if self.resource_check(pos, i, num):
                        self.current_resource_layer[pos[0], pos[1], i] += num
                        if (pos[0], pos[1]) in self.blank_pos:
                            self.blank_pos.remove((pos[0], pos[1]))
                    else:
                        raise ValueError(f"initial position {pos} and number {num} is not valid")
        
        for i in random_pos_resource_id:
            resource = self.Resource[i]
            random_list = self.blank_pos.copy()
            random.shuffle(random_list)
            count = -1
            for num in resource.init_num_list:
                while True:
                    count += 1
                    pos = random_list[count]
                    if self.resource_check(pos, i, num):
                        self.current_resource_layer[pos[0], pos[1], i] += num
                        self.blank_pos.remove((pos[0], pos[1]))
                        break
                    
                    if count >= len(random_list) - 1:
                        raise ValueError("No appropriate initial position!")
        
    
    def resource_check(self, pos, id, num):
        """Check the quantity of resource in a selected grid

        Args:
            pos: the coordinate of the resource
            id: the id of the resource
            num: the number of resource that is adding into the grid
        """

        return 0 <= self.current_resource_layer[pos[0], pos[1], id] + num \
            <= self.max_resource_layer[pos[0], pos[1], id]

    def all_resource_check(self, pos, num_array): 
        """Check the quantity all resources in a selected grid

        Args:
            pos: the coordinate of the resource
            num_array: the array of the resource num that are adding into the grid
        """

        tmp = self.current_resource_layer[pos[0], pos[1]] + num_array
        return (tmp >= 0).all() and (tmp <= self.max_resource_layer[pos[0], pos[1]]).all()
    
    def all_inventory_check(self, player_id: int, num_array):
        """Called when adding resources into players' inventories

        Args:
            player_id: the id of the player
            num_array: the array of resources num that are adding into the players'
                inventories
        """

        player = self.Player[player_id]
        tmp = player.inventory + num_array
        return (tmp >= 0).all() and (tmp <= player.inventory_max).all() \
            and np.sum(tmp) <= player.inventory_total_max
            
    def all_resource_requirement_check(self, player_id: int):
        """Check if the player's inventory meets the requirement
        
        Args:
            player_id (int): the id of the player
            
        Returns:
            bool: True if meet the requirement
        """
        player = self.Player[player_id]
        return np.all(player.inventory >= self.resource_requirement_array, axis=-1)
    
    def resource_requirement_check(self, player_id: int, resource_id: int):
        player = self.Player[player_id]
        return (player.inventory >= self.resource_requirement_array[resource_id]).all()
            
    def event_requirement_check(self, player_id: int, pos: tuple[int, int]) -> bool:
        """Check if the player's inventory meets the requirement

        Args:
            player_id (int): id of the player
            pos (tuple[int, int]): the position of the map

        Returns:
            bool: True if meet the requirement
        """
        
        player = self.Player[player_id]
        return (player.inventory >= self.event_requirement_layer[pos[0], pos[1]]).all()
    
    def event_available_check(self, pos: tuple[int, int]) -> bool:
        """Check if the event is available

        Args:
            pos (tuple[int, int]): the position of the map

        Returns:
            bool: True if the event exists and is available
        """
        return self.exist_event(pos) and self.event_avail_countdown_layer[pos[0], pos[1]] == 0
    
    def event_curriculum(self):
        if self.episodes % 5 == 0 and self.Event_num > 4:
            self.Event.pop()
            self.Event_num -= 1
            self.events_place = -np.ones((self.Event_num, 2), dtype=np.int16)
            
    def init_event_counter(self):
        H, W, C = self.event_io_layer.shape
        flatten_event_io = self.event_io_layer.reshape(H*W, C)
        unique_event_io = np.unique(flatten_event_io, axis=0)
        self.event_counter = {tuple(event_io): 0 for event_io in unique_event_io if (event_io != 0).any()}
        
    
    def update_event_counter(self, event_io):
        assert tuple(event_io) in self.event_counter, "Event IO not in the counter!"
        self.event_counter[tuple(event_io)] += 1
            
    def reset(self):
        """Reset an initial map"""
        self.steps = 0
        self.episodes += 1
        # self.event_curriculum()
        self.init_chunk()
        self.blank_pos = self.init_block()
        for player in self.Player:
            player.reset_inventory()
        self.init_player_pos()
        self.init_event_pos()
        self.init_resource_pos()
        self.init_extra_block()
        self.Communication.clear()
        self.social_state.reset()
        self.stage_controller.reset()
        
        self.history_credit = {
            i: self.Player[i].credit for i in self.Player_pool.alive_player_id
        }

        self.state_record = [
            {
                # 'state':self.chunk, 
                'inventory': {player.name: player.inventory for player in self.Player},
                # 'alive': {player.name: player.alive for player in self.Player},
            }
        ]
        self.stage_controller.update(self.state_record)
        
        self.init_event_counter()
        
        if self.With_Visual:
            self.update_tilechunk()

        return self.__obs__(self.Player_pool.alive_player_id)


    def step(self, action_dict: dict[str, Action]):
        """Update the map under some actions
        
        If more actions will be added, please update this function.

        Args:
            action_dict: A mapping that maps the name of the agent to its action
        """

        self.steps += 1
        self.Communication.clear()        
        temp_new_map: dict[tuple[int, int],list] = {}

        for name, action in action_dict.items():
            i = self.Player_pool.player_name2id[name]
            if isinstance(action, MovementAction):
                temp_new_location = (
                    (self.agents_place[i][0] + action.dx) % self.size[0],
                    (self.agents_place[i][1] + action.dy) % self.size[1]
                    )
                
                # Using a dict( (x,y) -> [agent_ids] ) to store agents' location
                if not (temp_new_location in temp_new_map):
                    temp_new_map[temp_new_location] = [i]
                else:
                    temp_new_map[temp_new_location].append(i)

            elif isinstance(action, Pick_Dump_Action):
                pos = self.agents_place[i]
                player = self.Player[i]
                if self.all_inventory_check(i, action.resources_vec):
                    if self.all_resource_check(pos, -action.resources_vec):
                        self.current_resource_layer[pos[0], pos[1]] -= action.resources_vec
                        player.inventory = player.inventory + action.resources_vec
                       
                        
            elif isinstance(action, Produce_Action):
                pos = self.agents_place[i]
                player = self.Player[i]
                if self.event_available_check(pos):
                    if self.event_requirement_check(i, pos):
                        if self.all_inventory_check(i, self.event_io_layer[pos[0], pos[1]]):
                            player.inventory = player.inventory + self.event_io_layer[pos[0], pos[1]]
                            self.update_event_counter(self.event_io_layer[pos[0], pos[1]])
                            self.load_event_countdown(pos)
                    
            elif isinstance(action, Communication_Action):
                self.Communication.load_message(i, action.towards, action.comm_vec)

            elif isinstance(action, SocialConnect_Action):
                if action.disconnect is not None:
                    self.social_state.remove(action.attribute, action.source, action.disconnect)
                if action.connect is not None:
                    self.social_state.set([action.attribute], action.source, action.connect, action.weights)

            elif isinstance(action, CreateSocialNode_Action):
                self.social_state.add_v(action.name, name, action.node_attribute)
                self.social_state.set(list(action.node_attribute.keys()), name, action.name)

            elif isinstance(action, RemoveSocialNode_Action):
                if self.default_social_graph[action.name]['creator'] == name:
                    self.social_state.remove_v(action.name)

        # Solve agent Collision, random pick one agent from each location to update
        # We first randomly choose one location to update, because later updated location will 
        # have advantage with less uncertainty, randomly shuffle the update make this as fair as possible
        
        if temp_new_map:
            temp_keys = list(temp_new_map.keys())
            random.shuffle(temp_keys)
            for key in temp_keys:
                if self.block_layer[key[0], key[1]] == 0:
                    if len(temp_new_map[key]) == 1:
                        agent_id_ = temp_new_map[key][0]
                    else:
                        agent_id_ = random.choice(temp_new_map[key])   
                    location = self.agents_place[agent_id_]

                    if self.agent_layer[key[0], key[1]] == 0:
                        self.agent_layer[key[0], key[1]] = agent_id_ + 1
                        self.agent_layer[location[0], location[1]] = 0
                        self.agents_place[agent_id_] = key

        # Update do-able Events
        # self.update_event_output(agent_influence, agent_influence_range)
        self.check_and_execute_event()

        if self.With_Visual:
            self.update_tilechunk()
        
        reward = self.__reward__(self.Player_pool.alive_player_id)
        
        self.state_record.append(
            {
                # 'state':self.chunk, 
                'inventory': {player.name: player.inventory for player in self.Player},
                # 'alive': {player.name: player.alive for player in self.Player},
                # 'reward': reward,
            }
        )
        self.stage_controller.update(self.state_record)

        # may change according to stage transition
        next_player_id = self.stage_controller.player_status(self.Player_pool)
        self.history_credit = {
            i: self.Player[i].credit for i in next_player_id
        }

        return self.__obs__(next_player_id), reward

    def get_subwindow(self, x_radius, y_radius, center_point):
        """Get a silce of subwindow for the current big map
        
        Args:
            x_radius: observation range in x-axis
            y_radius: observation range in y-axis
            center_point: the center point of the observation window
        """

        x, y = center_point
        rows = np.arange(x-x_radius, x+x_radius+1) % self.size[0]
        cols = np.arange(y-y_radius, y+y_radius+1) % self.size[1]
        return self.chunk[rows][:, cols]

    def __obs__(self, player_id_list: Sequence[int]) -> dict[str, Any]:
        """Get the observation of all agents whose indexes are in ``player_id_list``

        Args: 
            player_id_list: a list of agent ids
        """
        obs = {}
        for i in player_id_list:
            player = self.Player[i]
            obs[player.name] = {
                'grid_observation': self.get_subwindow(player.obs_range[0], player.obs_range[1], self.agents_place[i]),
                'inventory': player.inventory,
                'communication': self.Communication.board[:, i],
                'social_state': self.social_state.adj_matrix(DEFAULT_ATTRIBUTE, None),
                'time': np.array([self.steps]),
            }
        
        obs = self.stage_controller.obs_process(obs)
        obs = self.social_state.obs_process(obs)

        return obs
    
    def __reward__(self, player_id_list: Sequence[int]):
        """Get the reward of all agents whose indexes are in ``player_id_list``

        Args: 
            player_id_list: a list of agent ids.
        """
        reward = {}
        for i in player_id_list:
            player = self.Player[i]
            reward[player.name] = player.credit - self.history_credit[i]
        reward = self.stage_controller.reward_process(reward)
        reward = self.social_state.reward_process(reward)
        return reward
    
    def update_tilechunk(self):
        """Update tilechunk Map using Abstract_mapping.
        
        The rules under which circumstances which tile should be used are defined here.
        Can be rewrite if have different rules to show the current environment state.
        """

        self.tilechunk = np.zeros((self.size[0], self.size[1]), dtype=int)
        for i in range(self.size[0]):
            for j in range(self.size[1]):

                # We use the tile of the most had resources in a grid to represent it.

                resource_tile_vec = self.chunk[i, j, 2+self.Resource_num*2:2+self.Resource_num*3]
                if np.all(resource_tile_vec == 0):
                    pass
                else:
                    resource_tile_num = np.argmax(resource_tile_vec)
                    self.tilechunk[i, j] = self.Abstract_mapping["R_"+str(resource_tile_num)]

                event_tile_num = self.chunk[i, j, 1]
                if event_tile_num == -1:
                    pass
                else:
                    self.tilechunk[i, j] = self.Abstract_mapping["E_"+str(event_tile_num)]

                agent_tile_num = self.chunk[i, j, 0]
              
                if agent_tile_num == -1:
                    pass
                else:
                    self.tilechunk[i, j] = self.Abstract_mapping["A_"+str(agent_tile_num)]


    #WARNING: The setting of agent_influence is DEPRECATED now, which means the function may be buggy.
    # def update_event_output(self, agent_influence, agent_influence_range):
    #     """Update event's statues according to agent's influence

    #     Args:
    #         agent_influence: An Agent_num long list, each item is a Resource_num long array 
    #             which denotes influence on every possible resource output
    #         agent_influence_range: An Agent_num long list, each item is a number denotes 
    #             the radius one agent can influence
    #     """
    #     self.chunk[:, :, 2+self.Resource_num*3:2+self.Resource_num*4] = copy.deepcopy(self.chunk[:, :, 2:2+self.Resource_num])

    #     for i in range(len(agent_influence)):
    #         for j in range(-agent_influence_range[i], agent_influence_range[i]+1):
    #             for k in range(-agent_influence_range[i], agent_influence_range[i]+1):
    #                 x = (self.agents_place[i][0]+j) % self.size[0]
    #                 y = (self.agents_place[i][1]+k) % self.size[1]
    #                 if self.chunk[x, y, 1] != -1:
    #                     self.chunk[x, y, 2+self.Resource_num*3:2+self.Resource_num*4] = np.where(self.chunk[x, y, 2+self.Resource_num*3:2+self.Resource_num*4] > 0, np.maximum(
    #                         self.chunk[x, y, 2+self.Resource_num*3:2+self.Resource_num*4]+agent_influence[i], np.zeros(self.Resource_num)), self.chunk[x, y, 2+self.Resource_num*3:2+self.Resource_num*4])
    
    # If event's input resources is full, do event. Generate Output resources full.

    #TODO: add event consideration
    def check_and_execute_event(self):
        for pos in self.events_place:
            tmp = self.current_resource_layer[pos[0], pos[1]] + self.event_io_layer[pos[0], pos[1]]
            if (tmp>= 0).all() and (self.event_avail_countdown_layer[pos[0], pos[1]]) == 0:
                self.current_resource_layer[pos[0], pos[1]] = np.minimum(tmp, self.max_resource_layer[pos[0], pos[1]])
                self.update_event_counter(self.event_io_layer[pos[0], pos[1]])
                self.load_event_countdown(pos)
        
        self.event_avail_countdown_layer[:] = np.where(
            self.event_avail_countdown_layer > 0, self.event_avail_countdown_layer - 1, 0
        )

    #buggy now
    # def check_and_execute_event_with_event_range(self):
    #     for i in range(self.Event_num):
    #         Event_IO = self.chunk[self.events_place[i][0], self.events_place[i][1], 
    #                               2+self.Resource_num*3:2+self.Resource_num*4]
    #         Current_Resources = self.chunk[self.events_place[i][0], self.events_place[i][1],
    #                                        2+self.Resource_num*2:2+self.Resource_num*3]
    #         Max_Resources = self.chunk[self.events_place[i][0], self.events_place[i][1], 
    #                                    2+self.Resource_num:2+self.Resource_num*2]
    #         #print("====================",Current_Resources)
    #         #print("====================",Event_IO)
    #         if ((Current_Resources+Event_IO)>=0).all():
    #             #print("****************************************")
    #             self.chunk[self.events_place[i][0], self.events_place[i][1], 2+self.Resource_num*2:2 +
    #                        self.Resource_num*3] = Current_Resources-np.maximum(-Event_IO, 0)
    #             output_resource = np.maximum(Event_IO, 0)
    #             for j in range(self.Resource_num):
    #                 if output_resource[j]>0:
    #                     splited_resource=split_integer(output_resource[j],self.Event_random_output_range+1)
    #                     #print(splited_resource)
    #                     for k in range(self.Event_random_output_range+1):
                            
    #                         temp_total_ring=splited_resource[k]

    #                         if k==0:
    #                             self.chunk[self.events_place[i][0], self.events_place[i][1], 2+self.Resource_num*2+j] = np.minimum(self.chunk[self.events_place[i][0], self.events_place[i][1], 2+self.Resource_num*2+j]+temp_total_ring, self.chunk[self.events_place[i][0], self.events_place[i][1],2+self.Resource_num+j])
    #                         else:
    #                             #random_point=random.sample(range(0, temp_total_ring), temp_ring_blocks-1)
    #                             temp_ring_blocks=k*4
    #                             random_red_packet=get_random_red_packet(temp_total_ring,temp_ring_blocks)
    #                             random.shuffle(random_red_packet)
    #                             #print("====================",random_red_packet)

    #                             for ii in range(k):
    #                                 random1=random_red_packet[ii*4]
    #                                 random2=random_red_packet[ii*4+1]
    #                                 random3=random_red_packet[ii*4+2]
    #                                 random4=random_red_packet[ii*4+3]
    #                                 Max_Resources1=self.chunk[(self.events_place[i][0]+ii)% self.size[0], (self.events_place[i][1]+(k-ii))% self.size[1],
    #                                                           2+self.Resource_num+j]
    #                                 Max_Resources2=self.chunk[(self.events_place[i][0]-ii)% self.size[0], (self.events_place[i][1]-(k-ii))% self.size[1], 
    #                                                           2+self.Resource_num+j]
    #                                 Max_Resources3=self.chunk[(self.events_place[i][0]+ii)% self.size[0], (self.events_place[i][1]-(k-ii))% self.size[1], 
    #                                                           2+self.Resource_num+j]
    #                                 Max_Resources4=self.chunk[(self.events_place[i][0]-ii)% self.size[0], (self.events_place[i][1]+(k-ii))% self.size[1], 
    #                                                           2+self.Resource_num+j]
    #                                 self.chunk[(self.events_place[i][0]+ii)% self.size[0], (self.events_place[i][1]+(k-ii))% self.size[1], 2+self.Resource_num*2+j]=np.minimum(random1+self.chunk[(self.events_place[i][0]+ii)% self.size[0], (self.events_place[i][1]+(k-ii))% self.size[0], 2+self.Resource_num*2+j], Max_Resources1)
    #                                 self.chunk[(self.events_place[i][0]-ii)% self.size[0], (self.events_place[i][1]-(k-ii))% self.size[1], 2+self.Resource_num*2+j]=np.minimum(random2+self.chunk[(self.events_place[i][0]+ii)% self.size[0], (self.events_place[i][1]+(k-ii))% self.size[0], 2+self.Resource_num*2+j], Max_Resources2)
    #                                 self.chunk[(self.events_place[i][0]+ii)% self.size[0], (self.events_place[i][1]-(k-ii))% self.size[1], 2+self.Resource_num*2+j]=np.minimum(random3+self.chunk[(self.events_place[i][0]+ii)% self.size[0], (self.events_place[i][1]+(k-ii))% self.size[0], 2+self.Resource_num*2+j], Max_Resources3)
    #                                 self.chunk[(self.events_place[i][0]-ii)% self.size[0], (self.events_place[i][1]+(k-ii))% self.size[1], 2+self.Resource_num*2+j]=np.minimum(random4+self.chunk[(self.events_place[i][0]+ii)% self.size[0], (self.events_place[i][1]+(k-ii))% self.size[0], 2+self.Resource_num*2+j], Max_Resources4)
    
                                    #print("====================",random1)
                                    #os.system("pause")
    # Load a state from file.
    # TODO
    def load(self, path):
        pass

    # Save a current state to file.
    # TODO
    def save(self, path):
        pass

# Definition of a TileMap. Which load a tilechunk and turn it into a showable tilemap.


class TileMap:
    def __init__(self, tileset, background_tileset, tile_chunk, background_tilechunk, size=(20, 20), rect=None):
        self.size = size
        self.tileset = tileset
        self.tile_chunk = tile_chunk
        self.background_tileset=background_tileset
        self.background_tilechunk=background_tilechunk

        h, w = self.size
        self.image = pygame.Surface((tileset.size[0]*size[0], tileset.size[1]*size[1]))
        self.background_image = pygame.Surface((tileset.size[0]*size[0], tileset.size[1]*size[1]))

        if rect:
            self.rect = pygame.Rect(rect)
        else:
            self.rect = self.image.get_rect()

    def update(self, tile_chunk, background_tilechunk):
        self.tile_chunk = tile_chunk
        self.background_tilechunk=background_tilechunk

    # Render the TileMap, ready to be showed on screen.
    def render(self):

        #print(self.background_tilechunk)
        #print(self.tile_chunk)
        #os.system("pause")

        m, n = self.tile_chunk.shape
        for i in range(m):
            for j in range(n):
                tile = self.background_tileset.tiles[self.background_tilechunk[i, j]]
                self.background_image.blit(tile, (j*self.tileset.size[0], i*self.tileset.size[1]))

        self.image.blit(self.background_image, (0,0))
        
        for i in range(m):
            for j in range(n):
                tile = self.tileset.tiles[self.tile_chunk[i, j]]
                self.image.blit(tile, (j*self.tileset.size[0], i*self.tileset.size[1]))

    def __str__(self):
        return f'{self.__class__.__name__} {self.size}'
