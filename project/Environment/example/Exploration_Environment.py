# coding=UTF-8
# Here's the main file that execute the game, This file supports AI execution.
import os
from ..Map import Chunk, Big_Map, TileMap, MAX_ITEM_NUM
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete, Dict
from pygame.locals import *
from ..Game_Action import (
    Action, MovementAction, Pick_Dump_Action, Produce_Action, Communication_Action, 
    SocialConnect_Action, CreateSocialNode_Action, RemoveSocialNode_Action
    )
from ..Tileset import Tileset
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ..Social_State import SocialState, DEFAULT_ATTRIBUTE
from networkx import DiGraph
import types
GROUP = 'layer_1_{}'

def share_group_reward(cls: SocialState, group_num: int) -> None:
    def _reward_process(self: SocialState, reward: dict[str, float]):
        new_reward = reward.copy()
        for i in range(group_num):
            sum_reward = 0
            avail_player = []
            for player in self.G[DEFAULT_ATTRIBUTE].predecessors(GROUP.format(i)):
                if player in reward.keys():
                    sum_reward += reward[player]
                    avail_player.append(player)
            if len(avail_player) > 0:
                mean_reward = sum_reward / len(avail_player)
                for player in avail_player:
                    new_reward[player] = mean_reward
        return new_reward

    cls.reward_process = types.MethodType(_reward_process, cls)

def share_linked_obs(chunk: Chunk) -> None:
    player_id = chunk.Player_pool.player_name2id

    def _obs_process(self: SocialState, obs: dict[str, dict]):

        new_obs = obs.copy()
        sharing_info = {player: {} for player in obs.keys()}

        for player in obs.keys():
            # find relative location and id of observed agents from 'player'
            observed_players = dict()
            player_idx = (chunk.size[0]//2, chunk.size[1]//2)
            for i in range(obs[player]['grid_observation'].shape[1]):
                for j in range(obs[player]['grid_observation'].shape[2]):
                    if obs[player]['grid_observation'][0, i, j] > 0:
                        observed_players[obs[player]['grid_observation'][0, i, j]] = (i-player_idx[0], j-player_idx[1])
            # check edge relationships for sharing observations
            # for agent in self.G[DEFAULT_ATTRIBUTE].successors(player):
            #     if 'layer' not in agent and (player_id[agent]+1) in observed_players.keys():
            #         sharing_info[player][agent] = obs[agent]['grid_observation']
            for agent in self.G[DEFAULT_ATTRIBUTE].predecessors(player):
                if agent not in sharing_info[player].keys() and (player_id[agent]+1) in observed_players.keys():
                    sharing_info[player][agent] = obs[agent]['grid_observation']
            # aggregate the shared observation
            shared_obs = _assign_obs(chunk, sharing_info, player, obs[player]['grid_observation'], observed_players)
            new_obs[player]['grid_observation'] = shared_obs

        return new_obs

    def _assign_obs(chunk: Chunk, info: dict[str, dict], player: str, self_obs: np.array, observed_players: dict):
        shared_obs = np.copy(self_obs)

        for agent in info[player].keys():
            pos = observed_players[chunk.Player_pool.player_name2id[agent] + 1]
            obs = info[player][agent]

            for k in range(obs.shape[0]):
                for i in range(obs.shape[1]):
                    for j in range(obs.shape[2]):
                        if obs[k, i, j] > 0:
                            
                            info_idx = (i+pos[0], j+pos[1])
                            obs_idx = (info_idx[0]%chunk.size[0], info_idx[1]%chunk.size[1])
                            if shared_obs[k, obs_idx[0], obs_idx[1]] == 0:
                                shared_obs[k, obs_idx[0], obs_idx[1]] = obs[k, i, j]
        return shared_obs

    chunk.social_state.obs_process = types.MethodType(_obs_process, chunk.social_state)
    
def full_window(chunk: Chunk) -> None:
    def _get_window(self, x_radius, y_radius, center_point):
        x, y = center_point
        rows = np.arange(x-x_radius, x+x_radius+1) % self.size[0]
        cols = np.arange(y-y_radius, y+y_radius+1) % self.size[1]
        obs = self.chunk[rows][:, cols]
        new_obs = np.zeros((self.size[0], self.size[1], self.index.Layer_num), dtype=np.int32)
        player_idx = (self.size[0]//2, self.size[1]//2)
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                for l in range(self.index.Layer_num):
                    new_obs[player_idx[0]-x_radius+i, player_idx[1]-y_radius+j, l] = obs[i, j, l]
        return new_obs
    
    chunk.get_subwindow = types.MethodType(_get_window, chunk) 

class ID2Action:
    """Transfer the action id to action class"""
    def __init__(self, 
                 Resource_num: int, 
                 communication_dim: int,
                 group_num: int,
                 players: dict[int, str]) -> None:
        """Initialize the action id and action list

        Args:
            Resource_num (int): resource num of the env
            communication_dim (int): communication dimension of the env
            group_num (int): group num of the env
            players (dict[int, str]): dictionary of players' ids to names
            
        Attributes:
            action_list: a list of functions that take in the name of a player 
                and generate an Action
        """
        
        self.move = (0,5)
        self.produce = (5, 6)
        self.pick = (6, 6 + Resource_num)
        self.dump = (6 + Resource_num, 6 + 2*Resource_num)
        self.communication = (6 + 2*Resource_num,
                              6 + 2*Resource_num + communication_dim)
        self.connect_player = (6 + 2*Resource_num + communication_dim, 
                             6 + 2*Resource_num + communication_dim + len(players))
        self.attend_group = (6 + 2*Resource_num + communication_dim + len(players), 
                             6 + 2*Resource_num + communication_dim + len(players) + group_num)
        self.playerid2name = players
        dx_choice = [-1, 1, 0, 0, 0]; dy_choice = [0, 0, -1, 1, 0]
        resource_vecs = np.eye(Resource_num, dtype=np.int32)
        
        self.action_list = []
        for action_id in range(self.move[1] - self.move[0]):
            self.action_list.append(
                self.move_action(dx_choice[action_id], dy_choice[action_id])
                )
            
        for action_id in range(self.produce[1] - self.produce[0]):
            self.action_list.append(self.produce_action())
            
        for action_id in range(self.pick[1] - self.pick[0]):
            self.action_list.append(
                self.pick_dump_action(resource_vecs[action_id])
                )
            
        for action_id in range(self.dump[1] - self.dump[0]):
            self.action_list.append(
                self.pick_dump_action(-resource_vecs[action_id])
                )
            
        for action_id in range(self.communication[1] - self.communication[0]):
            tempzero=np.zeros((communication_dim,), dtype=np.float32)
            tempzero[action_id]=1
            self.action_list.append(
                self.communication_action(list(players.values()), tempzero)
                )
            
        for action_id in range(self.connect_player[1] - self.connect_player[0]):
            self.action_list.append(
                self.connect_player_action(action_id)
                )
        
        for action_id in range(self.attend_group[1] - self.attend_group[0]):
            self.action_list.append(
                self.social_connect_action(action_id)
                )
    
    def __call__(self, agent, graph, action_id):
        return self.action_list[action_id](agent, graph)
    
    def move_action(self, dx, dy):
        def _move(*args):
            return MovementAction(dx, dy)
        return _move
    
    def produce_action(self):
        def _produce(*args):
            return Produce_Action()
        return _produce
    
    def pick_dump_action(self, resource_vec):
        def _pick_dump(*args):
            return Pick_Dump_Action(resource_vec)
        return _pick_dump
    
    def communication_action(self, player, content):
        def _communicate(*args):
            return Communication_Action(player, content)
        return _communicate
    
    def connect_player_action(self, player_id):
        def _connect_player(agent, graph: DiGraph):
            linked = list(graph.successors(agent))
            target = self.playerid2name[player_id]
            if target in linked:
                return SocialConnect_Action(
                    DEFAULT_ATTRIBUTE,
                    agent,
                    connect=None,
                    disconnect=target
                )
            else:
                return SocialConnect_Action(
                    DEFAULT_ATTRIBUTE,
                    agent,
                    connect=target,
                    disconnect=None
                )
        return _connect_player
    
    def social_connect_action(self, action_id):
        def _social_connect(agent, graph: DiGraph):
            linked = list(graph.successors(agent))
            disconnect = None
            for node in linked:
                if node.startswith(GROUP):
                    disconnect = node
                    break
            return SocialConnect_Action(
                DEFAULT_ATTRIBUTE,
                agent,
                GROUP.format(action_id),
                disconnect
            )
        return _social_connect
        
        
class ExplorationEnv(MultiAgentEnv):

    def __init__(self, config):
        if config['With_Visual']==True:
            self.frames = []
            self.dts = []

        self.size = config['Map_size']
        self.render_fps=config['Render_fps']
        self.chunk = Chunk(Abstract_mapping = config['Abstract_mapping'], 
                           size = self.size, 
                           around_chunk = [], 
                           Block_num= config['Block_num'],
                           Block_pos=config['Block_pos'],
                           Player_feature = config['Player_feature'], 
                           Event_feature = config['Event_feature'],
                           Resource_feature = config['Resource_feature'],
                           grid_object_max = config['Grid_object_max'],
                           With_Visual = config['With_Visual'],
                           communication_length = config['Comm_words_dim'],
                           Social_node = config['Social_node'],
                           Social_attr = config['Social_attr'],
                           )

        self.step_num = 0
        self.terminated_point = config['Terminated_point']
        self.tile_margin=config['Tile_margin']
        self.tile_spacing=config['Tile_spacing']
        self.Resources_num = len(config['Resource_feature'])

        self.With_Visual=config['With_Visual']

        
        self.tile_root = config['Tile_root']
        self.background_tile_root=config['Background_tile_root']
        self.video_output_dir = config['Video_output_dir']
        
        self.record = config['record']
        self.record_dir = config['record_dir']
        
        # self.agent_influence = config['Agent_influence']
        # self.agent_influence_range = config['Agent_influence_range']
        self.players = self.chunk.player_name
        self.player_num = len(self.players)
        self.communication_words_dim = config["Comm_words_dim"]
        
        self.group_num = config["Group_num"]
        share_group_reward(self.chunk.social_state, self.group_num)
        share_linked_obs(self.chunk)
        full_window(self.chunk)
        action_dim = 2*self.Resources_num + 6 + self.communication_words_dim + self.player_num + self.group_num
        
        self.observation_space = {
            player.name: Dict({
                'grid_observation': Box(-MAX_ITEM_NUM, MAX_ITEM_NUM, 
                                        (2 + self.Resources_num*2, self.size[0], self.size[1])
                                        ),
                'inventory': Box(0, MAX_ITEM_NUM, (self.Resources_num,)),
                'communication': Box(0, 1, (self.player_num, config['Comm_words_dim'])),
                'social_state': Box(0, 1, (self.player_num + self.group_num, self.player_num + self.group_num)),
                'time': Box(0, self.terminated_point, (1,)),
                'player_id': Box(0, 1, (self.player_num + self.group_num,)),
                'action_mask': Box(0,1, (action_dim,))
            }) for player in self.chunk.Player
        }
        self.action_space = {player.name: Discrete(action_dim)
                             for player in self.chunk.Player}
        self.ID2Action = ID2Action(self.Resources_num, self.communication_words_dim, self.group_num, self.players)
        
        # if not os.path.exists(self.video_output_dir):
        #     os.makedirs(self.video_output_dir)

        if self.With_Visual:

            self.init_tiles(tile_size=config['Tile_size'])
            self.render_mode = config['Render_mode']

            pygame.init()

            self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)

            self.clock = pygame.time.Clock()
            self.running = True
            self.window_resize = None
    
    def clear_frame(self):
        self.frames=[]
        self.dts=[]

    def init_tiles(self, tile_size=(16, 16)):
        self.tile_size = tile_size
        self.H = self.size[0]*tile_size[0]
        self.W = self.size[1]*tile_size[1]
        self.window_size = self.H, self.W

        #print(self.tile_root)
        self.tileset = Tileset(self.tile_root,self.tile_size,self.tile_margin,self.tile_spacing)
        self.background_tileset = Tileset(self.background_tile_root,self.tile_size,self.tile_margin,self.tile_spacing)
        self.tileset.load()

    # Get info. Now pass.
    # TODO
    def _get_info(self):
        return (self.chunk.chunk,self.chunk.agents_place,self.chunk.events_place)
    
    def __obs__(self, player: str, observation):
        """final process to observation"""
        observation['action_mask'] = self.__actionmask__(player, observation["grid_observation"])
        id = self.chunk.index
        ind = list(range(id.Agent_ID[0], id.Agent_ID[1])) + \
        list(range(id.Block[0], id.Block[1])) + \
        list(range(id.Event_IO[0], id.Event_IO[1])) + \
        list(range(id.Cur_Res[0], id.Cur_Res[1])) # 1 + 1 + resource_num + resource_num
        observation["grid_observation"] = observation["grid_observation"].transpose((2, 0, 1))[ind]
        observation["time"] = self.terminated_point - observation["time"]
        observation["player_id"] = np.zeros((self.player_num + self.group_num, ))
        observation["player_id"][self.chunk.Player_pool.player_name2id[player]] = 1
        
        print(observation["social_state"])
        

        return observation
    
    def __actionmask__(self, player: str, grid_obs):
        """action mask for this environment"""
        action_mask = np.zeros((self.action_space[player].n, ))
        action_mask[self.ID2Action.move[0] : self.ID2Action.move[1]] = 1
        
        player_id = self.chunk.Player_pool.player_name2id[player]
        pos = self.chunk.agent_pos(player_id)
        
        if self.chunk.event_available_check(pos):
            if self.chunk.event_requirement_check(player_id, pos):
                if self.chunk.all_inventory_check(player_id, self.chunk.event_io_layer[pos[0], pos[1]]):
                    action_mask[self.ID2Action.produce[0] : self.ID2Action.produce[1]] = 1
        
        H, W, _ = grid_obs.shape
        pickable = grid_obs[H//2, W//2, self.chunk.index.Cur_Res[0]:self.chunk.index.Cur_Res[1]] > 0
        for id in range(self.Resources_num):
            if self.chunk.exist_resource_id(pos, id) and pickable[id]: # a little bit redunant
                action_mask[self.ID2Action.pick[0] + id] = 1
            if self.chunk.possess_resource(player, id):
                action_mask[self.ID2Action.dump[0] + id] = 1
                
        action_mask[self.ID2Action.communication[0]:self.ID2Action.communication[1]] = 1
        action_mask[self.ID2Action.attend_group[0]: self.ID2Action.attend_group[1]] = 1
        action_mask[self.ID2Action.connect_player[0]: self.ID2Action.connect_player[1]] = 1
        
        return action_mask

    # The reset function for gym.env calls the generate1 function in chunk.
    def reset(self, *, seed=None, options=None):
        obs = self.chunk.reset()
        self.step_num = 0

        # Using the tilechunk to define a tilemap for further rendering
        if self.With_Visual:
            self.tilemap = TileMap(self.tileset, self.background_tileset, self.chunk.tilechunk, 
                                   self.chunk.background_tilechunk, size=(self.size[0],self.size[1]))
            self.tilemap.render()

            self.image = self.tilemap.image
            self.rect = self.image.get_rect()

        self.render_handle()
        
        obs_output = self.obs_handle(obs)
        info_output = self.info_handle(obs)
        
        return obs_output, info_output


    def id2action(self, agent, action_id):
        return self.ID2Action(agent, self.chunk.default_social_graph, action_id)

    def extra_handle(self):
        if self.record:
            if self.step_num == 1:
                with open(self.record_dir, 'a') as file:
                    np.savetxt(file, self.chunk.event_layer[:, :, 0], delimiter = ',', fmt = "%d")
                    np.savetxt(file, self.chunk.current_resource_layer[:, :, 0], delimiter = ',', fmt = "%d")
                    np.savetxt(file, self.chunk.current_resource_layer[:, :, 1], delimiter = ',', fmt = "%d")
            with open(self.record_dir, 'a') as file:
                np.savetxt(file, np.array([[
                    player.credit for player in self.chunk.Player
                ]]), delimiter = ',', fmt = "%d")
            if self.step_num == self.terminated_point:
                with open(self.record_dir, 'a') as file:
                    np.savetxt(file, np.array([[
                        9999, 9999, 9999, 9999
                    ]]), delimiter = ',', fmt = "%d")
    
    def step(self, action_id): # @action_id: A dictionary {action name: action id}
        action = {agent: self.id2action(agent, action) for agent, action in action_id.items()}

        if self.With_Visual:
            self.tilemap.update(self.chunk.tilechunk, self.chunk.background_tilechunk)
            self.tilemap.render()

            self.image = self.tilemap.image
            self.rect = self.image.get_rect()
        
        obs, reward = self.chunk.step(action)        
        self.step_num += 1
        self.extra_handle()
        if (self.step_num >= self.terminated_point):
            truncated = {player: True for player in self.players}
            truncated['__all__'] = True

        else:
            truncated = {player: False for player in self.players}
            truncated['__all__'] = False

        terminated = truncated
        
        self.render_handle()
        
        obs_output = self.obs_handle(obs)
        reward_output = self.reward_handle(reward)
        info_output = self.info_handle(obs)

        return obs_output, reward_output, terminated, truncated, info_output
    
    def obs_handle(self, obs: dict[str, object]):
        return {
            player: self.__obs__(player, observation) for player,observation in obs.items()
        }
    
    def reward_handle(self, reward: dict[str, float]):
        return reward
        # mean_reward = sum(reward.values()) / len(reward)
        # return {player: mean_reward for player in reward.keys()}
    
    def info_handle(self, obs: dict):
        return {
            player: {} for player in obs.keys()
        }
        
    def render(self):
        '''
        if self.render_mode == "rgb_array":
            return self._render_frame()
        '''
        pass
    
    def render_handle(self):
        if self.With_Visual and self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):

        # Render a frame. Can do window resize.
        # Same as Pygame_test1.py
        for event in pygame.event.get():

            if event.type == QUIT:
                self.running = False

            elif event.type == VIDEORESIZE:

                self.window_resize = event.dict['size']

            '''
            elif event.type == KEYDOWN:

                action=[]
                for i in range(self.Agent_Num):
                    if event.key==K_UP:
                        action.append(MovementAction(-1,0))
                    elif event.key==K_DOWN:
                        action.append(MovementAction(1,0))
                    elif event.key==K_RIGHT:
                        action.append(MovementAction(0,1))
                    elif event.key==K_LEFT:
                        action.append(MovementAction(0,-1))

                self.chunk.update(action)
                self.tilemap.update(self.chunk.tilechunk)
                self.tilemap.render()
                
            else:
                pass
            '''

        if self.window_resize != None:
            self.screen.blit(pygame.transform.scale(self.image, self.window_resize), self.rect)
        else:
            self.screen.blit(self.image, self.rect)

        image_copy = self.image.copy()
        self.frames.append(image_copy)

        pygame.display.update()
        dt = self.clock.tick(self.render_fps)
        self.dts.append(dt)

    def close(self):
        if self.With_Visual:
            if self.screen is not None:
                pygame.display.quit()
                pygame.quit()

    def get_frames_and_dts(self):
        if self.With_Visual==True:
            return self.frames,self.dts
        else:
            return None,None
        
