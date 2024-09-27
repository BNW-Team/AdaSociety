from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np

class PPOProsocialPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        if other_agent_batches is not None:
            if "original_reward" not in sample_batch:
                sample_batch['original_reward'] = sample_batch[SampleBatch.REWARDS].copy()
            reward = sample_batch['original_reward']
            sum_reward =reward.copy()
            
            for _, (_, _, batch) in other_agent_batches.items():
                if "original_reward" not in batch:
                    batch["original_reward"] = batch[SampleBatch.REWARDS].copy()
                other_reward = batch["original_reward"]
                
                pad_reward = np.zeros_like(reward)
                ind = min(reward.shape[0], other_reward.shape[0])
                pad_reward[:ind] = other_reward[:ind]
                sum_reward += pad_reward
                
            mean_reward = sum_reward.astype(np.float32) / len(other_agent_batches)
            sample_batch[SampleBatch.REWARDS] = mean_reward

        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

