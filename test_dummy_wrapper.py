import torch
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

# 导入你自己的代码库函数
from constants import load_config, get_training_config, get_equipment_model, get_sim_task_config, DT
from sim_env import make_sim_env, BOX_POSE
from policy import ACTPolicy
from utils import sample_box_pose_eval
from ra_act_wrapper import RA_ACT_Wrapper

def test_wrapper_dummy():
    # 1. 加载配置
    config_path = "configs/fairino5_single_act_base/03_eval.yaml" 
    yaml_config = load_config(config_path)
    training_config = get_training_config(config_path)
    task_name = yaml_config.get('task', {}).get('name', 'sim_lifting_cube_scripted')
    equipment_model = get_equipment_model(config_path)
    ckpt_dir = training_config.get('ckpt_dir', './ckpts/fairino5_single_act_base')
    
    task_config = get_sim_task_config(task_name, config_path)
    camera_names = task_config['camera_names']
    
    # 2. 加载 stats
    with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)

    # 3. 初始化并加载冻结的 ACT 模型
    policy_config = {
        'num_queries': training_config.get('chunk_size', 100),
        'kl_weight': training_config.get('kl_weight', 10),
        'hidden_dim': training_config.get('hidden_dim', 512),
        'dim_feedforward': training_config.get('dim_feedforward', 3200),
        'lr_backbone': 1e-5, 'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
        'camera_names': camera_names, 'equipment_model': equipment_model
    }
    act_policy = ACTPolicy(policy_config)
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    act_policy.load_state_dict(torch.load(ckpt_path))
    act_policy.cuda()
    act_policy.eval()

    # 4. 初始化基础环境并包装
    base_env = make_sim_env(task_name, equipment_model)
    
    wrapper_config = {
        'camera_names': camera_names,
        'policy_config': policy_config,
        'state_dim': 7
    }
    
    # 构建 Wrapper
    rl_env = RA_ACT_Wrapper(base_env, act_policy, stats, wrapper_config)
    
    # 5. 开始 Dummy 测试
    num_rollouts = 50
    success_count = 0
    max_timesteps = 400
    
    for i in range(num_rollouts):
        BOX_POSE[0] = sample_box_pose_eval()
        obs = rl_env.reset()
        
        episode_reward = 0
        max_reward_in_ep = 0
        
        for t in range(max_timesteps):
            # 核心：RL 给出纯 0 动作，完全不干预 ACT
            zero_action = np.zeros(rl_env.action_space.shape)
            
            obs, reward, done, info = rl_env.step(zero_action)
            max_reward_in_ep = max(max_reward_in_ep, info['base_reward'])
            
            if done:
                break
                
        if max_reward_in_ep == base_env.task.max_reward:
            success_count += 1
            print(f"Rollout {i}: Success")
        else:
            print(f"Rollout {i}: Failed (Max Reward: {max_reward_in_ep})")
            
    print(f"\nDummy Test 成功率: {success_count}/{num_rollouts} = {success_count/num_rollouts*100}%")

if __name__ == '__main__':
    test_wrapper_dummy()