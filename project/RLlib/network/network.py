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
from collections import namedtuple
Graph = namedtuple('Graph', ['x', 'edge_index', 'batch'])

class TorchRNNModel(RecurrentNetwork, nn.Module):
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
        
        self.num_outputs = num_outputs
        #self.obs_size = get_preprocessor(obs_space)(obs_space).size

        self.fc_size = model_config["custom_model_config"].get("fc_size", 512)
        self.lstm_state_size = model_config["custom_model_config"].get("lstm_state_size", 256)

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.conv = nn.Sequential(
            nn.Conv2d(self.grid_obs_shape[0], 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(
            32 * self.grid_obs_shape[1] * self.grid_obs_shape[2]
            + self.inventory_shape[0]
            + self.communication_shape[0] * self.communication_shape[1]
            + self.social_state_shape[0] * self.social_state_shape[1]
            + self.time_shape[0]
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
        return torch.reshape(self.value_branch(self._features), [-1])

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

        communication = inputs["communication"]
        grid_observation = inputs["grid_observation"]
        inventory = inputs["inventory"]
        social_state = inputs["social_state"]
        time = inputs["time"]
        action_mask = inputs.get("action_mask", None)

        communication = communication.reshape(*(communication.shape[:-2]), -1)
        social_state = social_state.reshape(*(social_state.shape[:-2]), -1)
        x = grid_observation.reshape(
            grid_observation.shape[0]*grid_observation.shape[1], *self.grid_obs_shape
            )
        x = self.conv(x)
        x = x.reshape(grid_observation.shape[0], grid_observation.shape[1], -1)
        x = torch.cat((x, communication, inventory, social_state, time), dim = -1)
        x = self.fc1(x)
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        action = self.action_branch(self._features)
        if action_mask is not None:
            inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
            action = action + inf_mask
        return action, [torch.squeeze(h, 0), torch.squeeze(c, 0)]
    
class TorchCNNModel(TorchModelV2, nn.Module):
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
        
        self.fc1 = nn.Sequential(
            nn.Linear(
            32 * self.grid_obs_shape[1] * self.grid_obs_shape[2]
            + self.inventory_shape[0]
            + self.communication_shape[0] * self.communication_shape[1]
            + self.social_state_shape[0] * self.social_state_shape[1]
            + self.time_shape[0]
            , self.fc_size
            ),
            nn.ReLU(),
        )
        
        self.action_branch = nn.Linear(512, num_outputs)
        self.value_branch = nn.Linear(512, 1)
        
        self._features = None


    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        
        inputs = input_dict["obs"]
        communication = inputs["communication"]
        grid_observation = inputs["grid_observation"]
        inventory = inputs["inventory"]
        social_state = inputs["social_state"]
        time = inputs["time"]
        action_mask = inputs.get("action_mask", None)
        
        communication = communication.reshape(communication.shape[0], -1)
        social_state = social_state.reshape(social_state.shape[0], -1)

        x = self.conv(grid_observation)
        x = torch.cat((x, communication, inventory, social_state, time), dim = -1)
        self._features = self.fc1(x)
        action = self.action_branch(self._features)
        if action_mask is not None:
            inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
            action = action + inf_mask
        action = torch.reshape(action, [-1, self.num_outputs])
        return action, state
    

    
# class SocialPlayerModel(RecurrentNetwork, nn.Module):
#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name,
#     ):
#         nn.Module.__init__(self)
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)
#         self.grid_obs_shape = model_config["custom_model_config"]['grid_obs_shape']
#         self.social_obs_shape = model_config["custom_model_config"]['social_obs_shape']
#         self.type_shape = model_config["custom_model_config"]['type_shape']

#         self.breakpoint1 = -self.social_obs_shape[0] * self.social_obs_shape[1] - self.type_shape
#         self.breakpoint2 = -self.type_shape
        
#         self.num_outputs = num_outputs
#         #self.obs_size = get_preprocessor(obs_space)(obs_space).size

#         self.fc_size = model_config["custom_model_config"].get("fc_size", 256)
#         self.lstm_state_size = model_config["custom_model_config"].get("lstm_state_size", 256)

#         # Build the Module from fc + LSTM + 2xfc (action + value outs).
#         self.first_layer = nn.Conv2d(self.grid_obs_shape[0], 64, 3, 1, 1)
#         self.conv = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#         self.fc1 = nn.Linear(64*self.grid_obs_shape[1]*self.grid_obs_shape[2]+self.type_shape, self.fc_size)
#         self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)

#         self.social_process = nn.Sequential(
#             nn.Linear(self.social_obs_shape[0], 32),
#             nn.ReLU(),
#         )
#         self.get_social_feature = nn.Sequential(
#             nn.Linear(self.social_obs_shape[1] * 32, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#         )
        
#         self.action_branch = nn.Sequential(
#             nn.Linear(self.lstm_state_size + self.type_shape + 128, self.lstm_state_size),
#             nn.ReLU(),
#             nn.Linear(self.lstm_state_size, 5)
#         )
#         self.value_branch = nn.Sequential(
#             nn.Linear(self.lstm_state_size + self.type_shape + 128, self.lstm_state_size),
#             nn.ReLU(),
#             nn.Linear(self.lstm_state_size, 1)
#         )
        
#         # Build the Module from fc + LSTM + 2xfc (action + value outs).
#         # self.fc1 = nn.Linear(self.grid_obs_shape[0]*self.grid_obs_shape[1]*self.grid_obs_shape[2], 512)
#         # self.lstm = nn.LSTM(512, 512, num_layers=1,batch_first=True)
#         # self.action_branch = nn.Sequential(
#         #     nn.Linear(512,512),
#         #     nn.ReLU(),
#         #     nn.Linear(512,5),
#         # )
#         # self.value_branch = nn.Linear(512, 1)
#         # Holds the current "base" output (before logits layer).
#         self._features = None
#         #self.my_grid_place = [int(self.grid_obs_shape[1]//2), int(self.grid_obs_shape[2]//2)]
    
#     @override(ModelV2)
#     def get_initial_state(self):
#         # TODO: (sven): Get rid of `get_initial_state` once Trajectory
#         #  View API is supported across all of RLlib.
#         # Place hidden states on same device as model.
#         h = [
#             self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#             self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#         ]
#         return h

#     @override(ModelV2)
#     def value_function(self):
#         assert self._features is not None, "must call forward() first"
#         return torch.reshape(self.value_branch(self._features), [-1])

#     @override(RecurrentNetwork)
#     def forward(
#         self,
#         input_dict: Dict[str, TensorType],
#         state: List[TensorType],
#         seq_lens: TensorType,
#     ) -> Tuple[TensorType, List[TensorType]]:
#         """Adds time dimension to batch before sending inputs to forward_rnn().

#         You should implement forward_rnn() in your subclass."""
#         # Creating a __init__ function that acts as a passthrough and adding the warning
#         # there led to errors probably due to the multiple inheritance. We encountered
#         # the same error if we add the Deprecated decorator. We therefore add the
#         # deprecation warning here.
#         if log_once("recurrent_network_tf"):
#             deprecation_warning(
#                 old="ray.rllib.models.torch.recurrent_net.RecurrentNetwork"
#             )
#         # B: batch size
#         # input_dict["obs"] = {"grid_observation": Tensor with shape (B, _, _, _),
#         #     "self_inventory": Tensor with shape (B, _),
#         #     "communication": Tensor with shape (B, _)}
#         # input_dict["obs_flat"]: Tensor with shape (B, flatten and concatenate according to alphabet order of the key)
#         # e.g. input_dict["obs_flat"][:, self.communication_size:-self.inventory_size] == input_dict["obs"]["grid_observation"].flatten()

#         flat_inputs = input_dict["obs_flat"].float()
#         # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
#         # as input_dict may have extra zero-padding beyond seq_lens.max().
#         # Use add_time_dimension to handle this
#         self.time_major = self.model_config.get("_time_major", False)
#         inputs = add_time_dimension(
#             flat_inputs,
#             seq_lens=seq_lens,
#             framework="torch",
#             time_major=self.time_major,
#         )
#         output, new_state = self.forward_rnn(inputs, state, seq_lens)
#         output = torch.reshape(output, [-1, self.num_outputs])
#         return output, new_state

#     @override(RecurrentNetwork)
#     def forward_rnn(self, inputs, state, seq_lens):
        
#         """Feeds `inputs` (B x T x ..) through the Gru Unit.

#         Returns the resulting outputs as a sequence (B x T x ...).
#         Values are stored in self._cur_value in simple (B) shape (where B
#         contains both the B and T dims!).

#         Returns:
#             NN Outputs (B x T x ...) as sequence.
#             The state batches as a List of two items (c- and h-states).
#         """

#         obs = inputs[:, :, :self.breakpoint1].reshape(inputs.shape[0], inputs.shape[1],
#                            self.grid_obs_shape[0], self.grid_obs_shape[1], self.grid_obs_shape[2]).float()
#         social_obs = inputs[:, :, self.breakpoint1:self.breakpoint2].reshape(
#             inputs.shape[0], inputs.shape[1], self.social_obs_shape[0], self.social_obs_shape[1]).float()
#         types = inputs[:, :, self.breakpoint2: ].float()

#         x = self.first_layer(obs.reshape(inputs.shape[0] * inputs.shape[1],
#                            self.grid_obs_shape[0], self.grid_obs_shape[1], self.grid_obs_shape[2]))
#         x = self.conv(x)
#         x = x.reshape(inputs.shape[0], inputs.shape[1], x.shape[-1])
#         x = torch.cat((x, types), dim = -1)
#         x = nn.functional.relu(self.fc1(x))
#         obs_feature, [h, c] = self.lstm(
#             x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
#         )

#         social_obs = social_obs.permute(0, 1, 3, 2)
#         social_feature = self.social_process(social_obs)
#         social_feature = social_feature.reshape(social_feature.shape[0], social_feature.shape[1], -1)
#         social_feature = self.get_social_feature(social_feature)

#         self._features = torch.cat((obs_feature, types, social_feature), dim = -1)
#         move_action = self.action_branch(self._features)
#         return move_action, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

# class SocialPlayerCNNModel(TorchModelV2, nn.Module):
#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name,
#     ):
#         nn.Module.__init__(self)
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)
#         self.grid_obs_shape = model_config["custom_model_config"]['grid_obs_shape']
#         self.social_obs_shape = model_config["custom_model_config"]['social_obs_shape']
#         self.type_shape = model_config["custom_model_config"]['type_shape']
        
#         self.num_outputs = num_outputs

#         self.first_layer = nn.Conv2d(self.grid_obs_shape[0], 32, 3, 1, 1)
#         self.conv = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.Flatten()
#         )

#         self.social_process = nn.Linear(self.social_obs_shape[0], 32)
#         self.get_social_feature = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(self.social_obs_shape[1] * 32, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#         )

#         self.fc1 = nn.Linear(32*self.grid_obs_shape[1]*self.grid_obs_shape[2]+self.type_shape+128, 512)
        
#         self.action_branch = nn.Linear(512, num_outputs)
#         self.value_branch = nn.Linear(512, 1)

#         self._features = None
    
#     @override(ModelV2)
#     def value_function(self):
#         assert self._features is not None, "must call forward() first"
#         return torch.reshape(self.value_branch(self._features), [-1])

#     @override(RecurrentNetwork)
#     def forward(
#         self,
#         input_dict: Dict[str, TensorType],
#         state: List[TensorType],
#         seq_lens: TensorType,
#     ) -> Tuple[TensorType, List[TensorType]]:
#         obs = input_dict["obs"]["grid_observation"].float()
#         types = input_dict["obs"]["type"].float()
#         social_obs = input_dict["obs"]["social_obs"].float()
#         social_obs = social_obs.permute(0, 2, 1)

#         x = self.first_layer(obs)
#         x = self.conv(x)

#         social_x = self.social_process(social_obs)
#         social_x = social_x.reshape(social_x.shape[0], -1)
#         social_x = self.get_social_feature(social_x)

#         x = torch.cat((x, types, social_x), dim = -1)
#         self._features = nn.functional.relu(self.fc1(x))
#         output = self.action_branch(self._features)
#         output = torch.reshape(output, [-1, self.num_outputs])
#         return output, state

#     @override(RecurrentNetwork)
#     def forward_rnn(self, inputs, state, seq_lens):
        
#         """Feeds `inputs` (B x T x ..) through the Gru Unit.

#         Returns the resulting outputs as a sequence (B x T x ...).
#         Values are stored in self._cur_value in simple (B) shape (where B
#         contains both the B and T dims!).

#         Returns:
#             NN Outputs (B x T x ...) as sequence.
#             The state batches as a List of two items (c- and h-states).
#         """

#         obs = inputs[:, :, :self.breakpoint1].reshape(inputs.shape[0], inputs.shape[1],
#                            self.grid_obs_shape[0], self.grid_obs_shape[1], self.grid_obs_shape[2]).float()
#         social_obs = inputs[:, :, self.breakpoint1:self.breakpoint2].reshape(
#             inputs.shape[0], inputs.shape[1], self.social_obs_shape[0], self.social_obs_shape[1]).float()
#         types = inputs[:, :, self.breakpoint2: ].float()

#         x = self.first_layer(obs.reshape(inputs.shape[0] * inputs.shape[1],
#                            self.grid_obs_shape[0], self.grid_obs_shape[1], self.grid_obs_shape[2]))
#         x = self.conv(x)
#         x = x.reshape(inputs.shape[0], inputs.shape[1], x.shape[-1])
#         x = torch.cat((x, types), dim = -1)
#         x = nn.functional.relu(self.fc1(x))
#         obs_feature, [h, c] = self.lstm(
#             x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
#         )

#         social_obs = social_obs.permute(0, 1, 3, 2)
#         social_feature = self.social_process(social_obs)
#         social_feature = social_feature.reshape(social_feature.shape[0], social_feature.shape[1], -1)
#         social_feature = self.get_social_feature(social_feature)

#         self._features = torch.cat((obs_feature, types, social_feature), dim = -1)
#         move_action = self.action_branch(self._features)
#         return move_action, [torch.squeeze(h, 0), torch.squeeze(c, 0)]
    
# class SocialChoiceModel(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         nn.Module.__init__(self)
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)
#         self.obs_shape = model_config['custom_model_config']['obs_shape']
#         self.process = nn.Sequential(
#             nn.Linear(self.obs_shape[0], 32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU(),
#         )
#         self.action_head = nn.Sequential(
#             nn.Linear(self.obs_shape[1] * 32, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_outputs),
#         )
#         self.value_head = nn.Sequential(
#             nn.Linear(self.obs_shape[1] * 32, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#         )

#     def forward(self, input_dict, state, seq_lens):
#         obs = input_dict['obs'].float()
#         obs = obs.permute(0, 2, 1)
#         x = self.process(obs)
#         self._features = x.reshape([x.shape[0], -1])
#         output = self.action_head(self._features)
#         return output, []
    
#     def value_function(self) -> TensorType:
#         return self.value_head(self._features).reshape([-1])