from ray.rllib.env import MultiAgentEnv
from gymnasium.spaces import Box

def get_spaces_and_model_config(dummy_env: MultiAgentEnv, args):
    observation_space = dummy_env.observation_space
    action_space = dummy_env.action_space
    model_config_dict = {}
    for player, Dict_space in observation_space.items():
        model_config_dict[player] = {}
        dict_space = dict(Dict_space)
        for name, space in dict_space.items():
            assert isinstance(space, Box), 'space should be a gymnasium.Box'
            model_config_dict[player][name + '_shape'] = space.shape
        model_config_dict[player]['lstm_state_size'] = args.lstm_state_size
    return model_config_dict, observation_space, action_space