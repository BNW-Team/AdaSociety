from ..Environment.Social_State import DEFAULT_ATTRIBUTE
from datetime import datetime
from ..Environment import Event, Resource, Player
from .arg_utils import get_spaces_and_model_config
from ..Environment.example.Contract_Environment import GROUP, ContractEnv

def get_contract_complex_env_config(args):
     resource_list = [
          {
               Resource.NAME: "wood",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [5]*16,
          },
          {
               Resource.NAME: "stone",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [5]*4,
          },
          {
               Resource.NAME: "hammer",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [],
          },
          {
               Resource.NAME: "coal",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [5]*4,
          },
          {
               Resource.NAME: "torch",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [],
          },
          {
               Resource.NAME: "iron",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [2]*5,
          },
     ]
     player_list = [
          {
               Player.NAME: f"carpenter_{i}",
               Player.INIT_POS: None,
               Player.OBS_RANGE: (args.obs_range, args.obs_range),
               Player.ALIVE: True,
               Player.INITIAL_INVENTORY:{
                    "wood": 0,
                    "stone": 0, 
                    "hammer": 0,
                    'coal': 0,
                    'torch': 0,
                    'iron': 0,
               },
               Player.RESOURCE_VALUE: {
                    "wood": 1,
                    "stone": 1, 
                    "hammer": 5,
                    'coal': 10,
                    'torch': 30,
                    'iron': 20,
               },
               Player.INVENTORY_OBJECT_MAX: {
                    "wood": 100,
                    "stone": 100, 
                    "hammer": 1,
                    "coal": 0, 
                    'torch': 100,
                    'iron': 100,
               },
               Player.INVENTORY_TOTAL_MAX: 200,
          } for i in range(4)] + [
          {
               Player.NAME: f"miner_{i}",
               Player.INIT_POS: None,
               Player.OBS_RANGE: (args.obs_range, args.obs_range),
               Player.ALIVE: True,
               Player.INITIAL_INVENTORY:{
                    "wood": 0,
                    "stone": 0, 
                    "hammer": 0,
                    'coal': 0,
                    'torch': 0,
                    'iron': 0,
               },
               Player.RESOURCE_VALUE: {
                    "wood": 1,
                    "stone": 1,
                    "hammer": 5,
                    "coal": 10,
                    'torch': 30,
                    'iron': 20,
               },
               Player.INVENTORY_OBJECT_MAX: {
                    "wood": 100,
                    "stone": 0, 
                    "hammer": 100,
                    "coal": 100,
                    'torch': 1,
                    'iron': 0,
                    
               },
               Player.INVENTORY_TOTAL_MAX: 200,
          } for i in range(4)
     ]
     event_list = [
          {Event.NAME: "hammercraft"} for i in range(98)
     ]+[
          {Event.NAME: "torchcraft"} for i in range(98)
     ]
     env_config = {
          'Map_size': (args.size, args.size),
          'Render_fps': 4,
          'Abstract_mapping': {"A_0": 0, "A_1": 64, "A_2": 1574, "A_3": 1510, 
                         "E_0": 1214,
                         "R_0": 264, "R_1": 265, "R_2": 266},
          "Block_num": 0,
          "Block_pos": [],
          'Resource_feature': resource_list,
          'Player_feature': player_list,
          'Event_feature': event_list,
          'Grid_object_max': {"wood": 1000, "stone": 1000},
          'With_Visual': False,
          'Comm_words_dim':1,
          'Social_node': [player["name"] for player in player_list] \
               + [GROUP.format(i) for i in range(args.group_num)],
          'Social_attr': {
               DEFAULT_ATTRIBUTE: [],
          },
          'Terminated_point': args.terminated_point,
          'Tile_margin': 0,
          'Tile_spacing': 0,
          'Tile_size': 8,
          'Tile_root': None,
          'Background_tile_root': None,
          'Video_output_dir': None,
          'Render_mode': 'human',
          'Group_num': args.group_num,
          'Contract_round': args.contract_round,
          'contract_exploration_stage': args.contract_exploration_stage,
          'record': args.record,
          'record_dir': f'./results/record/ContractComplex_{args.algo}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
     }
     dummy_env = ContractEnv(env_config)
     (model_config_dict, 
      observation_space_dict, 
      action_space_dict) = get_spaces_and_model_config(dummy_env, args)
     for player in model_config_dict.keys():
          model_config_dict[player]['group_num'] = args.group_num
          model_config_dict[player]['player_num'] = len(player_list)
     
     return env_config, model_config_dict, observation_space_dict, action_space_dict
