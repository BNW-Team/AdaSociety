from ray.rllib.policy.policy import Policy
import numpy as np
from gymnasium.spaces import Tuple, Discrete, Dict
from ray.rllib.models.modelv2 import restore_original_dimensions
from itertools import accumulate
    
class RandomPolicy(Policy):
    def __init__(self,observation_space, action_space,config):
        super().__init__(observation_space, action_space,config)
        self.observation_space = observation_space
        
        if isinstance(self.observation_space.original_space, Dict) and \
            "action_mask" in dict(self.observation_space.original_space).keys():
            self.have_mask = True
        else:
            self.have_mask = False
        
        if isinstance(action_space, Tuple):
            self.num_output = [space.n for space in action_space]
        else:
            self.num_output = [action_space.n]
        self.accumulate_num_output = list(accumulate(self.num_output))
        self.accumulate_num_output.insert(0,0)
        
    def compute_actions(self,obs_batch,*args,**kwargs):
        action_batch=[]
        
        if self.have_mask:
            obs = restore_original_dimensions(obs_batch,self.observation_space,"numpy")
            action_mask = [
                obs["action_mask"][:, self.accumulate_num_output[i] : self.accumulate_num_output[i+1]]
                for i in range(len(self.accumulate_num_output)-1)
            ]
            for mask in action_mask:
                row_indices, col_indices = np.nonzero(mask)
                action_part = [
                    np.random.choice(col_indices[row_indices == i]) for i in range(obs_batch.shape[0])
                ]
            action_batch.append(action_part)
            action_batch = np.array(action_batch)
        else:
            for num in self.num_output:
                action_batch.append(np.random.randint(num, size=len(obs_batch)))
        
        action_batch = np.array(action_batch).T
        
        if len(self.num_output) == 1:
            action_batch = action_batch.squeeze(1)
        return action_batch,[],{}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass