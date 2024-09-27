from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import TensorType, try_import_torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util.debug import log_once
from typing import Dict, List, Union, Tuple
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
import torch
import torch.nn as nn
from .gnn_network import GNNModel
# torch, nn = try_import_torch()


class CentralizedCriticModel(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.grid_obs_shape = model_config["custom_model_config"]['grid_observation_shape']
        self.inventory_shape = model_config["custom_model_config"]['inventory_shape']
        self.communication_shape = model_config["custom_model_config"]['communication_shape']
        self.social_state_shape = model_config["custom_model_config"]['social_state_shape']
        self.time_shape = model_config["custom_model_config"]['time_shape']
        self.player_id_shape = model_config["custom_model_config"].get("player_id_shape", (0,))
        self.player_num = model_config["custom_model_config"].get("player_num", self.player_id_shape[0])
        self.group_num = model_config["custom_model_config"].get("group_num", None)
        self.with_player_id = False if self.player_id_shape[0] == 0 else True
        
        self.fc_size = model_config["custom_model_config"].get("fc_size", 512)
        
        
        self.num_outputs = num_outputs

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.conv = nn.Sequential(
            nn.Conv2d(self.grid_obs_shape[0], 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.social_embedding = GNNModel(64, True, model_config, self.with_player_id)
        
        if self.group_num is not None:
            self.social_choice = GNNModel(1, False, model_config, self.with_player_id)
        
        self.fc1 = nn.Sequential(
            nn.Linear(
            32 * self.grid_obs_shape[1] * self.grid_obs_shape[2]
            + self.inventory_shape[0]
            + self.communication_shape[0] * self.communication_shape[1]
            + 64
            + self.time_shape[0]
            + self.player_id_shape[0]
            , self.fc_size
            ),
            nn.ReLU(),
        )
        
        self.central_conv = nn.Sequential(
            nn.Conv2d(self.grid_obs_shape[0], 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
        )
        self.central_fc1= nn.Sequential(
            nn.Linear(
            16 * self.player_num * self.grid_obs_shape[1] * self.grid_obs_shape[2]
            + self.inventory_shape[0] * self.player_num
            + self.communication_shape[0] * self.communication_shape[1]
            + 64
            + self.time_shape[0]
            + self.player_id_shape[0]
            , self.fc_size
            ),
            nn.ReLU(),
        )
        
        self.action_branch = nn.Linear(self.fc_size, num_outputs)
        self.value_branch = nn.Linear(self.fc_size, 1)
        
        self._features = None

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self.value_branch(self._features), [-1])

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        my_obs = obs.unsqueeze(1)
        if opponent_obs.dim() == 2:
            return self.value_branch.weight.new(opponent_obs.shape[0], 1).zero_()
        opponent_obs_restore = restore_original_dimensions(opponent_obs, self.obs_space, "torch")
        my_obs_restore = restore_original_dimensions(my_obs, self.obs_space, "torch")
        all_grid_obs = torch.cat([my_obs_restore['grid_observation'], opponent_obs_restore['grid_observation']], dim = 1)
        B, T, C, H, W = all_grid_obs.shape
        x = self.central_conv(all_grid_obs.reshape(B*T, C, H, W)).reshape(B, -1)
        opponent_inventory_flatten = opponent_obs_restore['inventory'].reshape(B, -1)
        
        _, _, SS_H, SS_W = my_obs_restore["social_state"].shape
        social_state = my_obs_restore["social_state"].reshape(B, SS_H, SS_W)
        player_id = my_obs_restore["player_id"] if self.with_player_id else None
        social_embedding = self.social_embedding(social_state, player_id)
        if self.group_num is not None:
            self.social_choice(social_state, player_id)
            
        action_mask = my_obs_restore.get("action_mask", None)
        
        flatten_feature = [social_embedding]
        for key, value in my_obs_restore.items():
            if key not in ["grid_observation", "action_mask", "social_state"]:
                flatten_feature.append(value.reshape(B, -1))
                
        flatten_feature = torch.cat(flatten_feature, dim = -1)
        
        # print(x.shape)
        # print(self._flatten_features.shape)
        # print(opponent_inventory_flatten.shape)
        x = torch.cat((x, flatten_feature, opponent_inventory_flatten), dim = -1)
        x = self.central_fc1(x)
        phy_value = self.value_branch(x).reshape(-1)
        
        if self.group_num is not None:
            if self._action_mask is None:
                return self.social_choice.value_function() + phy_value
            else:
                action_mask = action_mask.squeeze(1)
                phy_factor = torch.any(action_mask[:, : -self.group_num] == 1, dim = -1)
                social_factor = torch.any(action_mask[:, -self.group_num : ] == 1, dim = -1)
                
                return phy_factor * phy_value + social_factor * self.social_choice.value_function()
        return phy_value
    
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        
        inputs = input_dict["obs"]
        grid_observation = inputs["grid_observation"]
        B, _, _, _ = grid_observation.shape
        
        action_mask = inputs.get("action_mask", None)
        self._action_mask = action_mask
        
        social_state = inputs["social_state"]
        player_id = inputs["player_id"] if self.with_player_id else None
        social_embedding = self.social_embedding(social_state, player_id)
        if self.group_num is not None:
            social_choice = self.social_choice(social_state, player_id)
        
        flatten_feature = [social_embedding]
        for key, value in inputs.items():
            if key not in ["grid_observation", "action_mask", "social_state"]:
                flatten_feature.append(value.reshape(B, -1))
                
        self._flatten_features = torch.cat(flatten_feature, dim = -1)

        x = self.conv(grid_observation)
        flatten_feature.append(x)
        x = torch.cat(flatten_feature, dim = -1)
        self._features = self.fc1(x)
        action = self.action_branch(self._features)
        
        if self.group_num is not None:
            action[:, -self.group_num: ] = social_choice[:, -self.group_num: ]
        if action_mask is not None:
            self.inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
            if action.shape[-1] == self.inf_mask.shape[-1]:
                action = action + self.inf_mask
        action = torch.reshape(action, [-1, self.num_outputs])
        return action, state

