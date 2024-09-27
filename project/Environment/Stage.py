import numpy as np
from .Player import PlayerPool
from .Map import Index

class StageController:
    def __init__(self, layer_index: Index, resource_requirement) -> None:
        self.stage_id = {}
        self.resource_requirement = resource_requirement
        self.layer_index = layer_index
    
    def reset(self):
        self.stage_id = {}
        
    def update(self, state_record):
        self.stage_id.update(state_record[-1]['inventory'])
        
    def obs_process(self, obs: dict):
        for player, player_obs in obs.items():
            resource_nonvisible = np.any(self.stage_id[player] < self.resource_requirement, axis = -1)
            resource_nonvisible = np.where(resource_nonvisible)[0]
            event_nonvisible = np.any(
                self.stage_id[player] < player_obs['grid_observation'][
                    :, :, self.layer_index.Event_requirement[0] : self.layer_index.Event_requirement[1]
                ],
                axis=-1
            )
            new_grid_obs = player_obs['grid_observation']
            new_grid_obs[:, :, resource_nonvisible + self.layer_index.Cur_Res[0]] = 0
            new_grid_obs[:, :, self.layer_index.Event_ID[0]:self.layer_index.Event_ID[1]][event_nonvisible] = 0
            new_grid_obs[:, :, self.layer_index.Event_avail_interval[0]:self.layer_index.Event_avail_interval[1]][event_nonvisible] = 0
            new_grid_obs[:, :, self.layer_index.Event_avail_countdown[0]:self.layer_index.Event_avail_countdown[1]][event_nonvisible] = 0
            new_grid_obs[:, :, self.layer_index.Event_IO[0]:self.layer_index.Event_IO[1]][event_nonvisible] = 0
            new_grid_obs[:, :, self.layer_index.Event_requirement[0]:self.layer_index.Event_requirement[1]][event_nonvisible] = 0
            obs[player]['grid_observation'] = new_grid_obs
        return obs
     
    def reward_process(self, reward):
        return reward

    def player_status(self, player_pool: PlayerPool):
        return player_pool.alive_player_id
    
    