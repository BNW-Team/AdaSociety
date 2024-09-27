# coding=UTF-8
# Here's the main file that execute the game, This file supports AI execution.
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



class ID2Action:
    def __init__(self, Resource_num, group_num) -> None:
        self.move = (0,5)
        self.produce = (5, 6)
        self.pick = (6, 6 + Resource_num)
        self.dump = (6 + Resource_num, 6 + 2*Resource_num)
        self.attend_group = (6 + 2*Resource_num, 6 + 2*Resource_num + group_num)

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


    def social_connect_action(self, action_id):
        def _social_connect(agent, graph: DiGraph):
            prev_group = list(graph.successors(agent))
            if len(prev_group) == 1:
                disconnect = prev_group[0]
            elif len(prev_group) == 0:
                disconnect = None
            else:
                raise AssertionError("A player can only connect to one group!")
            return SocialConnect_Action(
                DEFAULT_ATTRIBUTE,
                agent,
                GROUP.format(action_id),
                disconnect
            )
        return _social_connect


class ContractEnv(MultiAgentEnv):
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
        self.record_counter = 0

        self.players = self.chunk.player_name
        self.player_num = len(self.players)
        self.contract_time = config["Contract_round"] * self.player_num
        self.communication_words_dim = config["Comm_words_dim"]
        self.episode_num = 0

        # special in contract minitask
        self.group_num = config["Group_num"]
        share_group_reward(self.chunk.social_state, self.group_num)

        self.contract_exploration_stage = config.get("contract_exploration_stage", 3)

        self.observation_space = {
            player.name: Dict({
                'grid_observation': Box(
                    -MAX_ITEM_NUM,
                    MAX_ITEM_NUM,
                    (2+self.Resources_num*2, player.obs_range[0]*2 + 1, player.obs_range[1]*2 + 1),
                    dtype=np.int16
                ),
                'inventory': Box(0, MAX_ITEM_NUM, (self.Resources_num,), dtype=np.int16),
                'communication': Box(0, 1, (self.player_num, config['Comm_words_dim']), dtype=np.int8),
                'social_state': Box(0, 1, (self.player_num + self.group_num, self.player_num + self.group_num), dtype=np.int8),
                'time': Box(0, self.terminated_point + self.contract_time, (1,), dtype=np.int16),
                'player_id': Box(0, 1, (self.player_num + self.group_num,), dtype=np.int8),
                'action_mask': Box(0,1, (2*self.Resources_num + 6 + self.group_num,),dtype=np.int8)
            }) for player in self.chunk.Player
        }
        self.action_space = {player.name: Discrete(2*self.Resources_num + 6 + self.group_num)
                             for player in self.chunk.Player}
        self.ID2Action = ID2Action(self.Resources_num, self.group_num)

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
        id = self.chunk.index
        ind = list(range(id.Agent_ID[0], id.Agent_ID[1])) + \
        list(range(id.Block[0], id.Block[1])) + \
        list(range(id.Event_IO[0], id.Event_IO[1])) + \
        list(range(id.Cur_Res[0], id.Cur_Res[1])) # 1 + 1 + resource_num + resource_num
        observation["grid_observation"] = observation["grid_observation"].transpose((2, 0, 1))[ind]
        observation["time"] = self.terminated_point + self.contract_time - observation["time"]
        observation["player_id"] = np.zeros((self.player_num + self.group_num, ))
        observation["player_id"][self.chunk.Player_pool.player_name2id[player]] = 1
        observation['action_mask'] = self.__actionmask__(player)
        return observation

    def __actionmask__(self, player: str):
        """Action mask for this environment

        Args:
        player: str, the name of the player.

        Return: 
        An array of shape (action_space, ) for the player.
        0 represents the action is not available, 1 represents the action is available.
        
        0~4: Up, Down, Left, Right, Stay
        5: Produce
        6~(6+Resources_num): Pick Resource1, Pick Resource2, ...
        (6+Resources_num)~(6+2*Resources_num): Dump Resource1, Dump Resource2, ...
        (6+2*Resources_num)~(6+2*Resources_num+group_num): Attend Group1, Attend Group2, ...
        """

        action_mask = np.zeros((self.action_space[player].n, ))
        if self.step_num < self.contract_time:
            if self.contract_exploration_stage == 1 or self.contract_exploration_stage == 2: 
                action_mask[self.ID2Action.move[0]] = 1
            else:
                action_mask[self.ID2Action.attend_group[0]: self.ID2Action.attend_group[1]] = 1
            return action_mask

        action_mask[self.ID2Action.move[0] : self.ID2Action.move[1]] = 1

        player_id = self.chunk.Player_pool.player_name2id[player]
        pos = self.chunk.agent_pos(player_id)

        if self.chunk.event_available_check(pos):
            if self.chunk.event_requirement_check(player_id, pos):
                if self.chunk.all_inventory_check(player_id, self.chunk.event_io_layer[pos[0], pos[1]]):
                    action_mask[self.ID2Action.produce[0] : self.ID2Action.produce[1]] = 1

        pickable = self.chunk.all_resource_requirement_check(player_id)
        for id in range(self.Resources_num):
            if self.chunk.exist_resource_id(pos, id) and pickable[id]:
                action_mask[self.ID2Action.pick[0] + id] = 1
            if self.chunk.possess_resource(player, id):
                action_mask[self.ID2Action.dump[0] + id] = 1

        return action_mask

    def curriculum(self):
        self.fix_group = np.random.randint(self.ID2Action.attend_group[0], self.ID2Action.attend_group[1])
    
    # The reset function for gym.env calls the generate1 function in chunk.
    def reset(self, *, seed=None, options=None):
        obs = self.chunk.reset()
        self.episode_num += 1
        self.curriculum()
        self.step_num = 0
        self.phy_step_num = 0
        self.order = np.random.permutation(self.player_num)

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
        
        self.total_reward = np.zeros(self.player_num, dtype=np.float32)

        return obs_output, info_output


    def id2action(self, agent, action_id):
        return self.ID2Action(agent, self.chunk.default_social_graph, action_id)

    def extra_handle(self, reward_output):
        reward_this_step = []
        for key in sorted(self.players):
            reward_this_step.append(reward_output[self.players[key]] if self.players[key] in reward_output else 0)
        self.total_reward += np.array(reward_this_step)
        
        if self.step_num == self.terminated_point + self.contract_time:
            # print({player.name: player.inventory for player in self.chunk.Player})
            if self.record:
                with open(self.record_dir, 'a') as file:
                    np.savetxt(file, self.chunk.social_state.adj_matrix(), delimiter = ',')
                    np.savetxt(file, [ list(self.chunk.event_counter.values()) ], delimiter = ',')
                    np.savetxt(file, [[
                        player.credit for player in self.chunk.Player
                    ]], delimiter = ',', fmt = "%.1f")
                    np.savetxt(file, self.total_reward[np.newaxis, :], delimiter = ',', fmt = "%.1f")
                    np.savetxt(file, [[-1]], delimiter = ',')
                    self.record_counter += 1
                    print(self.record_counter)

    def before_action_handle(self, action_id: dict[str, int]):
        if self.step_num < self.contract_time:
            if self.contract_exploration_stage == 1:
                if self.step_num % self.player_num == 0:
                    self.proxy_action = np.random.randint(self.ID2Action.attend_group[0], self.ID2Action.attend_group[1])
                for player in action_id:
                    action_id[player] = self.proxy_action
            elif self.contract_exploration_stage == 2:
                for player in action_id:
                    action_id[player] = np.random.randint(self.ID2Action.attend_group[0], self.ID2Action.attend_group[1]) \
                        if np.random.rand() > 0.999**self.episode_num else self.fix_group
        return action_id

    def step(self, action_id): # @action_id: A dictionary {action name: action id}
        action_id = self.before_action_handle(action_id)
        action = {agent: self.id2action(agent, action) for agent, action in action_id.items()}

        if self.With_Visual:
            self.tilemap.update(self.chunk.tilechunk, self.chunk.background_tilechunk)
            self.tilemap.render()

            self.image = self.tilemap.image
            self.rect = self.image.get_rect()

        obs, reward = self.chunk.step(action)
        self.step_num += 1
        if (self.step_num >= self.terminated_point + self.contract_time):
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
        
        self.extra_handle(reward_output)

        return obs_output, reward_output, terminated, truncated, info_output

    def obs_handle(self, obs: dict):
        if self.step_num < self.contract_time:
            actor = self.players[
                self.order[int(self.step_num % self.player_num)]
            ]
            return {actor: self.__obs__(actor, obs[actor])}
        return {
            player: self.__obs__(player, observation) for player,observation in obs.items()
        }

    def reward_handle(self, reward: dict[str, float]):
        return reward


    def info_handle(self, obs: dict):
        return {}

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

