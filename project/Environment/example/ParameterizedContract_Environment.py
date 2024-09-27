# coding=UTF-8
# Here's the main file that execute the game, This file supports AI execution.
import os
from ..Map import Chunk, Big_Map, TileMap, MAX_ITEM_NUM
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete, Dict
from pygame.locals import *
from ..Game_Action import (
    Action, MovementAction, Pick_Dump_Action, Communication_Action, 
    SocialConnect_Action, CreateSocialNode_Action, RemoveSocialNode_Action
    )
from ..Tileset import Tileset
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ..Social_State import SocialState, DEFAULT_ATTRIBUTE
import types

GROUP = 'layer_1_{}'
PORTION = [0, 0.2, 0.5]
MOVEMENT = [(0,0), (1,0), (0,1), (-1,0), (0,-1)]

class Action_ID:
    def __init__(self, Resource_num, Player_num) -> None:
        self.move = (0,5)
        self.pick_dump = (5, 5 + 2*Resource_num)
        self.create_group = (5 + 2*Resource_num, 8 + 2*Resource_num)
        self.attend_group = (8 + 2*Resource_num, 8 + 2*Resource_num + Player_num)


class ParameterizedContractEnv(MultiAgentEnv):
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
        
        self.players = self.chunk.player_name
        self.player_num = len(self.players)
        self.communication_words_dim = config["Comm_words_dim"]

        self.observation_space = {
            player.name: Dict({
                'grid_observation': Box(
                    -MAX_ITEM_NUM, 
                    MAX_ITEM_NUM, 
                    (2+self.Resources_num*2, player.obs_range[0]*2 + 1, player.obs_range[1]*2 + 1)
                ),
                'inventory': Box(0, MAX_ITEM_NUM, (self.Resources_num,)),
                'communication': Box(0, 1, (self.player_num, config['Comm_words_dim'])),
                'social_state': Box(0, 1, (self.player_num, self.player_num)),
                'time': Box(0, self.terminated_point, (1,)),
                'action_mask': Box(0,1, (2*self.Resources_num + 8 + self.player_num,))
            }) for player in self.chunk.Player
        }
        self.action_space = {player.name: Discrete(2*self.Resources_num + 8 + self.player_num)
                             for player in self.chunk.Player}
        self.Action_ID = Action_ID(self.Resources_num, self.player_num)

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

    @property
    def player_name2id(self):
        return self.chunk.Player_pool.player_name2id
    
    # Get info. Now pass.
    # TODO
    def _get_info(self):
        return (self.chunk.chunk,self.chunk.agents_place,self.chunk.events_place)
    
    def __obs__(self, player, observation):
        id = self.chunk.index
        ind = list(range(id.Agent_ID[0], id.Agent_ID[1])) + \
        list(range(id.Block[0], id.Block[1])) + \
        list(range(id.Event_IO[0], id.Event_IO[1])) + \
        list(range(id.Cur_Res[0], id.Cur_Res[1])) # 1 + 1 + resource_num + resource_num
        observation["grid_observation"] = observation["grid_observation"].transpose((2, 0, 1))[ind]
        
        observation['social_state'] = self.social_obs
        observation['action_mask'] = self.__actionmask__(player)
        return observation
    
    def __actionmask__(self, player):
        action_mask = np.zeros((2*self.Resources_num + 8 + self.player_num,))
        if self.step_num == 0:
            action_mask[self.Action_ID.move[0]] = 1
            action_mask[self.Action_ID.create_group[0] : self.Action_ID.create_group[1]] = 1
        elif self.step_num == 1:
            if player in self.creator_list:
                action_mask[self.Action_ID.move[0]] = 1
            else:
                action_mask[self.Action_ID.attend_group[0] : self.Action_ID.attend_group[1]] = self.attend_action_mask
        else:
            action_mask[self.Action_ID.move[0] : self.Action_ID.move[1]] = 1
            action_mask[self.Action_ID.pick_dump[0] : self.Action_ID.pick_dump[1]] = 1

        return action_mask

    # The reset function for gym.env calls the generate1 function in chunk.
    def reset(self, *, seed=None, options=None):
        obs = self.chunk.reset()
        self.step_num = 0
        self.social_obs = np.zeros((self.player_num, self.player_num), np.float32)

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
        if action_id < 5:
            action = MovementAction(*(MOVEMENT[action_id]))
        elif action_id < 2*self.Resources_num + 5:
            res_vec = np.zeros((self.Resources_num,), dtype=np.int32)
            res_vec[(action_id - 5) // 2] = ((action_id - 5) % 2) * 2 - 1
            action = Pick_Dump_Action(resources_vec=res_vec)
        elif action_id < 2*self.Resources_num + 8:
            action = CreateSocialNode_Action(
                name=GROUP.format(agent),
                node_attribute={
                    DEFAULT_ATTRIBUTE: {
                        'portion': PORTION[action_id - (2*self.Resources_num + 5)]
                    }
                }
            )
        else:
            action = SocialConnect_Action(
                DEFAULT_ATTRIBUTE, 
                agent,
                GROUP.format(
                  self.chunk.player_name[action_id - (2*self.Resources_num + 8)]
                ),
                None
            )
        return action
    
    def make_social_obs_dict(self):
        """create the social obs in a certain episode"""
        dict_view = {}
        for node in self.group_list:
            node_attr = self.chunk.default_social_graph.nodes[node]
            row = self.player_name2id[node_attr['creator']]
            portion = node_attr['portion']
            dict_view[row] = {}
            cols = [player for player in self.chunk.default_social_graph.predecessors(node)]
            for player in cols:
                dict_view[row][player] = 1/len(cols) * (1 - portion)
            dict_view[row][node_attr['creator']] += portion
        
        return dict_view
    
    def make_social_process(self, cls: SocialState, social_obs: dict[int, dict[str, float]]) -> None:
        """Alternate reward_process() in SocialState according to contracts"""
        def _reward_process(self: SocialState, reward: dict[str, float]):
            new_reward = reward.copy()
            for group in social_obs.values():
                total_reward = sum(
                    [reward[player] for player in group.keys()]
                )
                for player, portion in group.items():
                    new_reward[player] = portion * total_reward
            return new_reward
        
        cls.reward_process = types.MethodType(_reward_process, cls)
    
    def social_obs_dict2array(self, dict_view: dict[int, dict[str, float]]):
        """transfer the social obs from dictionary to array"""
        social_obs_array = np.zeros((self.player_num, self.player_num), np.float32)
        for row, group in dict_view.items():
            for player, portion in group.items():
                social_obs_array[row][self.player_name2id[player]] = portion
        return social_obs_array

    def extra_handle(self):
        """handle extra process in ``step()``"""
        if self.step_num == 1:
            self.group_list = [
                node for node in self.chunk.default_social_graph.nodes
                    if isinstance(node, str) and node.startswith("layer_1")
            ]
            self.creator_list = [
                self.chunk.default_social_graph.nodes[node]["creator"] for node in self.group_list
            ]
            self.attend_action_mask = np.zeros((self.player_num,))
            self.attend_action_mask[
                [self.player_name2id[player] for player in self.creator_list]
            ] = 1
        
        elif self.step_num == 2:
            social_obs_dict_view = self.make_social_obs_dict()
            self.social_obs = self.social_obs_dict2array(social_obs_dict_view)
            self.make_social_process(self.chunk.social_state, social_obs_dict_view)
            if self.record:
                with open(self.record_dir, 'a') as file:
                    np.savetxt(file, self.social_obs, delimiter = ',', fmt = "%.1f")
                    
        elif self.step_num == self.terminated_point:
            if self.record:
                with open(self.record_dir, 'a') as file:
                    np.savetxt(file, [
                        player.credit for player in self.chunk.Player
                    ], delimiter = ',', fmt = "%.1f")
        
    def step(self, action_id):
        action = {agent: self.id2action(agent, action) for agent, action in action_id.items()}

        if self.With_Visual:
            self.tilemap.update(self.chunk.tilechunk, self.chunk.background_tilechunk)
            self.tilemap.render()

            self.image = self.tilemap.image
            self.rect = self.image.get_rect()
        
        obs, reward_output = self.chunk.step(action)   
        
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
        info_output = self.info_handle(obs)

        return obs_output, reward_output, terminated, truncated, info_output
    
    def obs_handle(self, obs: dict):
        return {
            player: self.__obs__(player, observation) for player,observation in obs.items()
        }
        
    def info_handle(self, obs: dict):
        return {
            player : {} for player in obs.keys()
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
        