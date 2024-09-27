from ..Environment.Social_State import DEFAULT_ATTRIBUTE
from datetime import datetime
from ..Environment import Event, Resource, Player
from ..Environment.example.Exploration_Environment import ExplorationEnv
from .arg_utils import get_spaces_and_model_config


def get_basic_env_config(args):
    resource_list = [
        {
            Resource.NAME: "wood",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [10]*10,
        },
        {
            Resource.NAME: "stone",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [10]*10,
        },
        {
            Resource.NAME: "hammer",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "coal",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [5]*10,
        },
        {
            Resource.NAME: "torch",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [],
        },
        {
            Resource.NAME: "iron",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [4]*10,
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
            Resource.INIT_NUM_LIST: [2]*5,
        },
        {
            Resource.NAME: "clay",
            Resource.INIT_POS_LIST: None,
            Resource.INIT_NUM_LIST: [4]*10,
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
                    "power": 0,
                    "hammer": 0,
                    "coal": 0,
                    "torch": 0,
                    "iron": 0,
                    "steel": 0,
                    "shovel": 0,
                    "pickaxe": 0,
                    "gem_mine": 0,
                    "clay": 0,
                    "pottery": 0,
                    "cutter": 0,
                    "gem": 0,
                    "totem": 0
            },
            Player.RESOURCE_VALUE: {
                    "wood": 1,
                    "stone": 1,
                    "power": 0,
                    "hammer": 5,
                    "coal": 2,
                    "torch": 8,
                    "iron": 3,
                    "steel": 10,
                    "shovel": 40,
                    "pickaxe": 50,
                    "gem_mine": 4,
                    "clay": 4,
                    "pottery": 20,
                    "cutter": 30,
                    "gem": 70,
                    "totem": 500
            },
            Player.INVENTORY_OBJECT_MAX: {
                    "wood": 20,
                    "stone": 20,
                    "power": 10,
                    "hammer": 20,
                    "coal": 20,
                    "torch": 20,
                    "iron": 20,
                    "steel": 20,
                    "shovel": 20,
                    "pickaxe": 20,
                    "gem_mine": 20,
                    "clay": 20,
                    "pottery": 20,
                    "cutter": 20,
                    "gem": 20,
                    "totem": 20
            },
            Player.INVENTORY_TOTAL_MAX: 200,
        } for i in range(0, 2)
    ]
    event_list = [
        {
            Event.NAME: f"woodwork_req{i}",
            Event.INIT_POS: None,
            Event.INPUT: {"wood": 1, "stone": 1},
            Event.OUTPUT: {"axe": 1},
            Event.REQUIREMENT: {"power": 1},
            Event.AVAIL_INTERVAL: 1
        } for i in range(45)
    ] + [{
              Event.NAME: f"wood_origin{i}",
              Event.INIT_POS: None,
              Event.INPUT: {},
              Event.OUTPUT: {"wood": 1},
              Event.REQUIREMENT: {},
              Event.AVAIL_INTERVAL: 3
         } for i in range(2)
    ] + [{
              Event.NAME: f"stone_origin{i}",
              Event.INIT_POS: None,
              Event.INPUT: {},
              Event.OUTPUT: {"stone": 1},
              Event.REQUIREMENT: {},
              Event.AVAIL_INTERVAL: 3
         } for i in range(2)
    ]
    env_config = {
        'Map_size': (args.size, args.size),
        'Render_fps': 4,
        'Abstract_mapping': {"A_0": 1, "A_1": 65, "A_2": 1575, "A_3": 1511, 
                         "E_0": 189, "E_1": 1773, "E_2": 1644, "E_3": 250, "E_4": 1770, "E_5": 1771, "E_6": 1257, "E_7": 1482, "E_8": 1722,
                         "R_0": 902, "R_1": 2554, "R_2": 192, "R_3": 928, "R_4": 1187, "R_5": 1303, "R_6": 1237, "R_7": 1190, "R_8": 1191, "R_9": 1063, "R_10": 2510, "R_11": 1750, "R_12": 2438, "R_13": 1050, "R_14": 738},
        "Block_num": 0,
        "Block_pos": [],
        'Resource_feature': resource_list,
        'Player_feature': player_list,
        'Event_feature': event_list,
        'Grid_object_max': {
                "wood": 1000,
                "stone": 1000,
                "power": 1000,
                "hammer": 1000,
                "coal": 1000,
                "torch": 1000,
                "iron": 1000,
                "steel": 1000,
                "shovel": 1000,
                "pickaxe": 1000,
                "gem_mine": 1000,
                "clay": 1000,
                "pottery": 1000,
                "cutter": 1000,
                "gem": 1000,
                "totem": 1000
            },
        'With_Visual': False,
        'Comm_words_dim':1,
        'Social_node': [player["name"] for player in player_list],
        'Social_attr': {
            DEFAULT_ATTRIBUTE: [],
        },
        'Terminated_point': args.terminated_point,
        'Tile_margin': 0,
        'Tile_spacing': 0,
        'Tile_size': 48,
        'Tile_root': None,
        'Background_tile_root': './resources/cutted_tileset',
        'Video_output_dir': None,
        'Render_mode': 'human',
        'record': args.record,
        'record_dir': f'./results/record/Basic{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    }
    dummy_env = ExplorationEnv(env_config)
    (model_config_dict, 
    observation_space_dict, 
    action_space_dict) = get_spaces_and_model_config(dummy_env)

    return env_config, model_config_dict, observation_space_dict, action_space_dict