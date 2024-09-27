from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import TensorType
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util.debug import log_once
from typing import Dict, List, Union, Tuple
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.nn as gnn
from collections import namedtuple
import math
Graph = namedtuple('Graph', ['x', 'edge_index', 'batch'])

class TorchGRNNModel(RecurrentNetwork, nn.Module):
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
        self.social_state_shape = model_config["custom_model_config"]['social_state_shape']
        self.communication_shape = model_config["custom_model_config"]['communication_shape']
        self.player_id_shape = model_config["custom_model_config"].get("player_id_shape", (0,))
        self.with_player_id = False if self.player_id_shape[0] == 0 else True
        
        shape_num = 0
        for key, value in model_config["custom_model_config"].items():
            if key.endswith("_shape") and key not in ["grid_observation_shape", 
                                                      "social_state_shape", 
                                                      "communication_shape",
                                                      "player_id_shape",
                                                      "action_mask_shape"]:
                shape_num += math.prod(value)
        
        self.group_num = model_config["custom_model_config"].get("group_num", None)
        
        
        self.num_outputs = num_outputs
        #self.obs_size = get_preprocessor(obs_space)(obs_space).size

        self.fc_size = model_config["custom_model_config"].get("fc_size", 512)
        self.lstm_state_size = model_config["custom_model_config"].get("lstm_state_size", 128)

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.conv = nn.Sequential(
            nn.Conv2d(self.grid_obs_shape[0], 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.social_embedding = GNNModel(64, True, model_config, self.with_player_id)
        
        if self.group_num is not None:
            self.social_choice = GNNModel(1, False, model_config, self.with_player_id)
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(
            32 * self.grid_obs_shape[1] * self.grid_obs_shape[2]
            + self.communication_shape[0] * self.communication_shape[1]
            + 64
            + self.player_id_shape[0]
            + shape_num
            , self.fc_size
            ),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        
        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        # self.fc1 = nn.Linear(self.grid_obs_shape[0]*self.grid_obs_shape[1]*self.grid_obs_shape[2], 512)
        # self.lstm = nn.LSTM(512, 512, num_layers=1,batch_first=True)
        # self.action_branch = nn.Sequential(
        #     nn.Linear(512,512),
        #     nn.ReLU(),
        #     nn.Linear(512,5),
        # )
        # self.value_branch = nn.Linear(512, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None
        #self.my_grid_place = [int(self.grid_obs_shape[1]//2), int(self.grid_obs_shape[2]//2)]
    
    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        phy_value = torch.reshape(self.value_branch(self._features), [-1])
        if self.group_num is not None:
            if self._action_mask is None:
                return self.social_choice.value_function() + phy_value
            else:
                self._action_mask = self._action_mask.reshape(
                    self._action_mask.shape[0] * self._action_mask.shape[1], -1
                    )
                phy_factor = torch.any(self._action_mask[:, : -self.group_num] == 1, dim = -1)
                social_factor = torch.any(self._action_mask[:, -self.group_num : ] == 1, dim = -1)
                return phy_factor * phy_value + social_factor * self.social_choice.value_function()
        return phy_value

    @override(RecurrentNetwork)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        # Creating a __init__ function that acts as a passthrough and adding the warning
        # there led to errors probably due to the multiple inheritance. We encountered
        # the same error if we add the Deprecated decorator. We therefore add the
        # deprecation warning here.
        if log_once("recurrent_network_tf"):
            deprecation_warning(
                old="ray.rllib.models.torch.recurrent_net.RecurrentNetwork"
            )
        # B: batch size
        # input_dict["obs"] = {"grid_observation": Tensor with shape (B, _, _, _),
        #     "self_inventory": Tensor with shape (B, _),
        #     "communication": Tensor with shape (B, _)}
        # input_dict["obs_flat"]: Tensor with shape (B, flatten and concatenate according to alphabet order of the key)
        # e.g. input_dict["obs_flat"][:, self.communication_size:-self.inventory_size] == input_dict["obs"]["grid_observation"].flatten()

        flat_inputs = input_dict["obs_flat"].float()
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        rnn_input = input_dict["obs"].copy()
        for key, value in rnn_input.items():
            rnn_input[key] = add_time_dimension(
            value,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )
        # inputs = add_time_dimension(
        #     flat_inputs,
        #     seq_lens=seq_lens,
        #     framework="torch",
        #     time_major=self.time_major,
        # )
        output, new_state = self.forward_rnn(rnn_input, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """

        grid_observation = inputs["grid_observation"]
        B, T, _, _, _ = grid_observation.shape
        
        action_mask = inputs.get("action_mask", None)
        self._action_mask = action_mask
        
        social_state = inputs["social_state"]
        player_id = inputs["player_id"] if self.with_player_id else None
        social_state = social_state.reshape(B*T, *self.social_state_shape)
        social_embedding = self.social_embedding(social_state, player_id).reshape(B, T, -1)
        if self.group_num is not None:
            social_choice = self.social_choice(social_state, player_id).reshape(B, T, -1)
        
        flatten_feature = [social_embedding]
        for key, value in inputs.items():
            if key not in ["grid_observation", "action_mask", "social_state"]:
                flatten_feature.append(value.reshape(B, T, -1))
        
        x = grid_observation.reshape(B*T, *self.grid_obs_shape)
        x = self.conv(x)
        x = x.reshape(B, T, -1)
        flatten_feature.append(x)
        x = torch.cat(flatten_feature, dim = -1)
        x = self.fc1(x)
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        
        action = self.action_branch(self._features)
        if self.group_num is not None:
            action[:,:, -self.group_num: ] = social_choice[:, :, -self.group_num: ]
        if action_mask is not None:
            inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
            action = action + inf_mask
        return action, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

class TorchGCNNModel(TorchModelV2, nn.Module):
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
        self.group_num = model_config["custom_model_config"].get("group_num", None)
        self.with_player_id = False if self.player_id_shape[0] == 0 else True
        
        self.fc_size = model_config["custom_model_config"].get("fc_size", 256)
        
        
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
        
        self.action_branch = nn.Linear(self.fc_size, num_outputs)
        self.value_branch = nn.Linear(self.fc_size, 1)
        
        self._features = None


    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        phy_value = torch.reshape(self.value_branch(self._features), [-1])
        if self.group_num is not None:
            if self._action_mask is None:
                return self.social_choice.value_function() + phy_value
            else:
                phy_factor = torch.any(self._action_mask[:, : -self.group_num] == 1, dim = -1)
                social_factor = torch.any(self._action_mask[:, -self.group_num : ] == 1, dim = -1)
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
    
class GNNModel(nn.Module):
    def __init__(self, num_outputs, global_pool, model_config, with_player_id):
        super().__init__()
        self.social_state_shape = model_config["custom_model_config"]['social_state_shape']
        self.hidden_dim = model_config["custom_model_config"].get("hidden_dim", 128)
        self.fc_size = model_config["custom_model_config"].get("fc_size", 512)
        self.global_pool = global_pool
        
        self.num_outputs = num_outputs
        
        if with_player_id:
            input_shape = self.social_state_shape[0] + 1
        else:
            input_shape = self.social_state_shape[0]

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        if global_pool:
            self.actor_gnn = gnn.Sequential('x, edge_index, batch', [
            (GCNConv(input_shape, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (global_mean_pool, 'x, batch -> x'),
            nn.Linear(64, num_outputs)
        ])
        
        else:
            self.actor_gnn = gnn.Sequential('x, edge_index, batch', [
                (GCNConv(input_shape, 64), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
                (GCNConv(64, 64), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
                (GCNConv(64, 64), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_outputs)
            ])
        
        self.value_gnn = gnn.Sequential('x, edge_index, batch', [
            (GCNConv(input_shape, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (global_mean_pool, 'x, batch -> x'),
            nn.Linear(64, 1)
        ])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"


    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_gnn(self._features.x, self._features.edge_index, self._features.batch), [-1])

    def forward(self, social_state, player_id) -> Tuple[TensorType, List[TensorType]]:
        B, h, w = social_state.shape
        node = torch.eye(h).expand(B, -1, -1).to(self.device)
        connect = torch.where((social_state + node) >= 1)
        
        node = node.reshape(B*h, w).float()
        if player_id is not None:
            player_id = player_id.reshape(B*h, 1).float()
            node = torch.cat([node, player_id], dim = -1)
        edge = torch.stack((connect[1], connect[2]))
        edge += connect[0] * h

        batch = torch.arange(B).repeat_interleave(h).to(self.device)
        self._features = Graph(node, edge.to(torch.long), batch)
        
        action = self.actor_gnn(self._features.x, self._features.edge_index, self._features.batch)
        if not self.global_pool:
            action = action.reshape(B, h * self.num_outputs)
        return action
