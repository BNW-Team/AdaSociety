from ..Environment.Social_State import DEFAULT_ATTRIBUTE
from datetime import datetime
from ..Environment import Event, Resource, Player
from ..Environment.example.Exploration_Environment import ExplorationEnv, GROUP
from .arg_utils import get_spaces_and_model_config


def get_exploration_env_config(args):
     resource_list = [
          {
            Resource.NAME: "wood",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [20]*10,
        },
        {
            Resource.NAME: "stone",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [20]*10,
        },
        {
            Resource.NAME: "hammer",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "coal",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [10]*10,
        },
        {
            Resource.NAME: "torch",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "iron",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [8]*10,
        },
        {
            Resource.NAME: "steel",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "shovel",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "pickaxe",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "gem_mine",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [4]*5,
        },
        {
            Resource.NAME: "clay",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [8]*10,
        },
        {
            Resource.NAME: "pottery",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "cutter",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "gem",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "totem",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
          
     ]
     player_list = [
          {
               Player.NAME: f"player_{i}",
               Player.INIT_POS: None,
               Player.OBS_RANGE: (args.obs_range, args.obs_range),
               Player.ALIVE: True,
               Player.INITIAL_INVENTORY:{},
               Player.RESOURCE_VALUE: {
                    "wood": 1,
                    "stone": 1,
                    "hammer": 5,
                    "coal": 2,
                    "torch": 20,
                    "iron": 3,
                    "steel": 30,
                    "shovel": 100,
                    "pickaxe": 150,
                    "gem_mine": 4,
                    "clay": 4,
                    "pottery": 40,
                    "cutter": 100,
                    "gem": 200,
                    "totem": 1000
            },
               Player.INVENTORY_OBJECT_MAX: {},
               Player.INVENTORY_TOTAL_MAX: 32767,
          } for i in range(0, 8)
     ]
     event_list = [
          {Event.NAME: "hammercraft"} for i in range(40)
     ]+[
          {Event.NAME: "torchcraft"} for i in range(40)
     ]+[
          {Event.NAME: "steelmaking"} for i in range(30)
     ]+[
          {Event.NAME: "potting"} for i in range(30)
     ]+[
          {Event.NAME: "shovelcraft"} for i in range(20)
     ]+[
          {Event.NAME: "pickaxecraft"} for i in range(20)
     ]+[
          {Event.NAME: "cuttercraft"} for i in range(20)
     ]+[
          {Event.NAME: "gemcutting"} for i in range(10)
     ]+[
          {Event.NAME: "totemmaking"} for i in range(10)
     ]
     env_config = {
          'Map_size': (args.size, args.size),
          'Render_fps': 4,
          'Abstract_mapping': [],
          "Block_num": 25,
          "Block_pos": [],
          'Resource_feature': resource_list,
          'Player_feature': player_list,
          'Event_feature': event_list,
          'Grid_object_max': {},
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
          "Group_num": args.group_num,
          'record': args.record,
          'record_dir': f'./results/record/Exploration{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
     }
     dummy_env = ExplorationEnv(env_config)
     (model_config_dict, 
      observation_space_dict, 
      action_space_dict) = get_spaces_and_model_config(dummy_env, args)
     for player in model_config_dict.keys():
          model_config_dict[player]['player_num'] = len(player_list)
     
     return env_config, model_config_dict, observation_space_dict, action_space_dict