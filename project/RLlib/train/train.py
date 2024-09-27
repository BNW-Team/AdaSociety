import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import numpy as np
from datetime import datetime
from itertools import count
from tqdm import tqdm
import copy
from ..network import TorchRNNModel, TorchCNNModel, TorchGRNNModel, TorchGCNNModel, CentralizedCriticModel
from typing import Callable
# from Global import GlobalVar

import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.r2d2 import R2D2Config, R2D2TorchPolicy
from ..policy import RandomPolicy, PPOProsocialPolicy, CCPPOTorchPolicy, DQNMaskTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ...Environment.example import ContractEnv, ExplorationEnv, ParameterizedContractEnv
from ...args import (get_contract_env_config, 
                     get_exploration_env_config, 
                     get_parameterized_contract_env_config,
                     get_contract_complex_env_config)

ModelCatalog.register_custom_model("cnn_model", TorchCNNModel)
ModelCatalog.register_custom_model("lstm_model", TorchRNNModel)
ModelCatalog.register_custom_model("gcnn_model", TorchGCNNModel)
ModelCatalog.register_custom_model("glstm_model", TorchGRNNModel)
ModelCatalog.register_custom_model("centralized_model", CentralizedCriticModel)


def var_handle(task_name: str) -> tuple[str, Callable]:
    if task_name == "contract":
        env_name = "ContractEnv"
        get_env_config = get_contract_env_config
        register_env(env_name, lambda config: ContractEnv(config))
        
    elif task_name == "exploration":
        env_name = "ExplorationEnv"
        get_env_config = get_exploration_env_config
        register_env(env_name, lambda config: ExplorationEnv(config))

    elif task_name == 'parameterized_contract':
        env_name = "ParameterizedContract"
        get_env_config = get_parameterized_contract_env_config
        register_env(env_name, lambda config: ParameterizedContractEnv(config))
        
    elif task_name == 'contract_complex':
        env_name = "ContractEnv"
        get_env_config = get_contract_complex_env_config
        register_env(env_name, lambda config: ContractEnv(config))
        
    return env_name, get_env_config

def train(args):
    """contract training function"""
    env_name, get_env_config = var_handle(args.task)
    ray.init()
    config, model_config_dict, obs_space_dict, action_space_dict = get_env_config(args)

    # global_var = GlobalVar.options(name = "record").remote(args.total_stages)

    if args.gnn:
        if args.lstm:
            player_model_name = "glstm_model"
        else:
            player_model_name = "gcnn_model"
    else:
        if args.lstm:
            player_model_name = "lstm_model"
        else:
            player_model_name = "cnn_model"

    if args.algo == 'Rainbow':
        if args.lstm:
            algo_config = R2D2Config()
            policy_name = R2D2TorchPolicy
        else:
            algo_config = DQNConfig()
            policy_name = DQNMaskTorchPolicy
        algo_name = 'rainbow'
    elif args.algo == 'PPO':
        algo_config = PPOConfig()
        policy_name = PPOTorchPolicy
        algo_name = 'ppo'
    elif args.algo == 'random':
        algo_config = A3CConfig()
        policy_name = RandomPolicy
        algo_name = 'random'
    elif args.algo == 'PPOProsocial':
        algo_config = PPOConfig()
        policy_name = PPOProsocialPolicy
        algo_name = 'ppo_prosocial'
    elif args.algo == 'CCPPO':
        algo_config = PPOConfig()
        policy_name = CCPPOTorchPolicy
        algo_name = 'ccppo'
        player_model_name = 'centralized_model'

    name_1 = config["Player_feature"][0]["name"]
    algo_config = (
        algo_config
        .environment(env_name, env_config=config)
        .framework("torch")
        .rollouts(num_rollout_workers=args.num_rollout_workers, 
                  num_envs_per_worker=args.num_envs_per_worker, 
                  rollout_fragment_length=args.rollout_fragment_length)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=1,)
        .rl_module( _enable_rl_module_api=False)
        .training(gamma = args.gamma,
                  lr = args.lr,
                    model = {
                        "max_seq_len": args.max_seq_len,
                        "custom_model": player_model_name,
                        "custom_model_config": model_config_dict[name_1]
                    },
                    train_batch_size = args.num_rollout_workers * 
                        args.num_envs_per_worker * args.rollout_fragment_length,
                    _enable_learner_api=False,)
    )
    # extra config for different algorithms
    if args.algo == 'Rainbow':
        dqn_model_config_dict = algo_config["model"]
        # dqn_model_config_dict["no_final_linear"] = True
        algo_config = (
            algo_config
            .rollouts(compress_observations=True)
            .training(
                num_steps_sampled_before_learning_starts = args.num_cold_start_steps,
                num_atoms = args.num_atoms,
                v_min = args.v_min,
                v_max = args.v_max,
                noisy = args.noisy, 
                n_step = args.n_step,
                model = dqn_model_config_dict,
                )
        )
        # if args.lstm:
        #     algo_config = algo_config.training(num_atoms = 1)
    elif args.algo == 'PPO' or args.algo == 'PPOProsocial':
        algo_config = (
            algo_config.training(
                sgd_minibatch_size = args.sgd_minibatch_size,
                num_sgd_iter = args.num_sgd_iter,
                grad_clip = args.grad_clip,
                )
        )

    algo_config_list = []
    for player in config['Player_feature']:
        new_model_config_dict = algo_config["model"]
        new_model_config_dict["custom_model_config"] = model_config_dict[player["name"]]
        algo_config_list.append(
            algo_config.training(model = new_model_config_dict)
        )

    policies = {
        f'{algo_name}_{player["name"]}': PolicySpec(
            policy_name, 
            obs_space_dict[player["name"]], 
            action_space_dict[player["name"]], 
            algo_config_list[i]
        ) for i, player in enumerate(config['Player_feature'])
    }
    for i, player in enumerate(config['Player_feature']):
        player_type = player["name"].split('_')[0]
        if f"{algo_name}_{player_type}" not in policies:
            policies[f"{algo_name}_{player_type}"] = copy.deepcopy(policies[f'{algo_name}_{player["name"]}'])

    if args.share:
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            agent_type = agent_id.split('_')[0]
            return f"{algo_name}_{agent_type}"
    else:
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return f"{algo_name}_{agent_id}"
        
    if args.algo == 'random':
        policies_to_train = []
    else:
        policies_to_train = list(policies.keys())
    algo_config = algo_config.multi_agent(
        policies = policies,
        policy_mapping_fn = policy_mapping_fn,
        policies_to_train = policies_to_train,
    )

    algo = algo_config.build()
    if args.checkpoint != '':
        algo.restore(args.checkpoint)
    for i in range(args.max_training_iter):
        result = algo.train()
        result['sampler_results']['hist_stats'] = None
        result['info'] = None
        print(pretty_print(result))
        if (i+1)% args.save_interval == 0:
            path = algo.save()
            #torch.save(ppo_algo.get_policy('ppo').get_weights(), config["save_dir"] + "stage{stage}.pth")
            print(f"Checkpoint loaded in {path}")
    algo.stop()
