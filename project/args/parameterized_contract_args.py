from ..Environment.Social_State import DEFAULT_ATTRIBUTE
from datetime import datetime
from ..Environment import Event, Resource, Player
from ..Environment.example.ParameterizedContract_Environment import ParameterizedContractEnv
from .arg_utils import get_spaces_and_model_config


def get_parameterized_contract_env_config(args):
     resource_list = [
          {
               Resource.NAME: "wood",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [10]*2,
          },
          {
               Resource.NAME: "stone",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [10]*2,
          },
          {
               Resource.NAME: "axe",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [],
          },
          {
               Resource.NAME: "power",
               Resource.INIT_POS_LIST: None,
               Resource.INIT_NUM_LIST: [],
          }
     ]
     player_list = [
          {
               Player.NAME: f"player_{i}",
               Player.INIT_POS: None,
               Player.OBS_RANGE: (args.size // 2, args.size // 2),
               Player.ALIVE: True,
               Player.INITIAL_INVENTORY:{
                    "wood": 0,
                    "stone": 0, 
                    "axe": 0,
                    "power": 1,
               },
               Player.RESOURCE_VALUE: {
                    "wood": 1,
                    "stone": 1, 
                    "axe": 5,
                    "power": 0,
               },
               Player.INVENTORY_OBJECT_MAX: {
                    "wood": 20,
                    "stone": 20, 
                    "axe": 1,
                    "power": 10,
               },
               Player.INVENTORY_TOTAL_MAX: 200,
          } for i in range(0, 2)
     ] + [
          {
               Player.NAME: f"player_{i}",
               Player.INIT_POS: None,
               Player.OBS_RANGE: (args.size // 2, args.size // 2),
               Player.ALIVE: True,
               Player.INITIAL_INVENTORY:{
                    "wood": 0,
                    "stone": 0, 
                    "axe": 0,
                    "power": 0,
               },
               Player.RESOURCE_VALUE: {
                    "wood": 1,
                    "stone": 1, 
                    "axe": 10,
                    "power": 0,
               },
               Player.INVENTORY_OBJECT_MAX: {
                    "wood": 0,
                    "stone": 0, 
                    "axe": 100,
                    "power": 0,
               },
               Player.INVENTORY_TOTAL_MAX: 200,
          } for i in range(2, 4)
     ]
     event_list = [
          {
               Event.NAME: f"woodwork_req{i}",
               Event.INIT_POS: None,
               Event.INPUT: {"wood": 1, "stone": 1},
               Event.OUTPUT: {"axe": 1},
               Event.REQUIREMENT: {"power": 1},
               Event.AVAIL_INTERVAL: 1
          } for i in range(42)
     ]
     env_config = {
          'Map_size': (args.size, args.size),
          'Render_fps': 4,
          'Abstract_mapping': [],
          "Block_num": 3,
          "Block_pos": [],
          'Resource_feature': resource_list,
          'Player_feature': player_list,
          'Event_feature': event_list,
          'Grid_object_max': {"wood": 1000, "stone": 1000},
          'With_Visual': False,
          'Comm_words_dim':2,
          'Social_node': [player["name"] for player in player_list],
          'Social_attr': {
               DEFAULT_ATTRIBUTE: [] 
          },
          'Terminated_point': args.terminated_point,
          'Tile_margin': 0,
          'Tile_spacing': 0,
          'Tile_size': 8,
          'Tile_root': None,
          'Background_tile_root': None,
          'Video_output_dir': None,
          'record': args.record,
          'record_dir': f'./results/record/exp{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
          'Render_mode':'human',
     }
     dummy_env = ParameterizedContractEnv(env_config)
     (model_config_dict, 
      observation_space_dict, 
      action_space_dict) = get_spaces_and_model_config(dummy_env, args)
     for player in model_config_dict.keys():
          model_config_dict[player]['player_num'] = len(player_list)
     
     return env_config, model_config_dict, observation_space_dict, action_space_dict