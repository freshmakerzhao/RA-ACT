import os
import torch
import pickle
import numpy as np
from stable_baselines3 import SAC

from constants import load_config, get_training_config, get_equipment_model, get_sim_task_config
from sim_env import make_sim_env
from policy import ACTPolicy
from ra_act_wrapper import RA_ACT_Wrapper

def evaluate_rl_success_rate():
    config_path = "configs/fairino5_single_act_base/03_eval.yaml" 
    yaml_config = load_config(config_path)
    training_config = get_training_config(config_path)
    
    task_name = yaml_config.get('task', {}).get('name', 'sim_lifting_cube_scripted')
    equipment_model = get_equipment_model(config_path)
    ckpt_dir = training_config.get('ckpt_dir', './ckpts')
    task_config = get_sim_task_config(task_name, config_path)
    camera_names = task_config['camera_names']
    
    with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)

    policy_config = {
        'num_queries': training_config.get('chunk_size', 100),
        'kl_weight': training_config.get('kl_weight', 10),
        'hidden_dim': training_config.get('hidden_dim', 512),
        'dim_feedforward': training_config.get('dim_feedforward', 3200),
        'lr_backbone': 1e-5, 'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
        'camera_names': camera_names, 'equipment_model': equipment_model
    }
    act_policy = ACTPolicy(policy_config)
    act_policy.load_state_dict(torch.load(os.path.join(ckpt_dir, 'policy_best.ckpt')))
    act_policy.cuda()
    act_policy.eval()

    base_env = make_sim_env(task_name, equipment_model)
    wrapper_config = {
        'task_name': task_name,
        'equipment_model': equipment_model,
        'camera_names': camera_names,
        'policy_config': policy_config,
        'state_dim': 14 if 'bimanual' in equipment_model else (4 if 'excavator' in equipment_model else 7)
    }
    rl_env = RA_ACT_Wrapper(base_env, act_policy, stats, wrapper_config, epsilon=0.08)

    model_path = "./sac_ra_act_logs/ra_act_sac_final" 
    model = SAC.load(model_path)

    num_rollouts = 50
    success_count = 0
    max_timesteps = 400
    
    print("🚀 开始在 OOD 区域进行批量评估 (50次)...")
    for i in range(num_rollouts):
        obs = rl_env.reset()
        max_reward_in_ep = 0
        
        for t in range(max_timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = rl_env.step(action)
            max_reward_in_ep = max(max_reward_in_ep, info['base_reward'])
            if done:
                break
                
        if max_reward_in_ep == base_env.task.max_reward:
            success_count += 1
            print(f"Rollout {i}: 成功")
        else:
            print(f"Rollout {i}: 失败 (Max Reward: {max_reward_in_ep})")
            
    print(f"\n✅ 结合残差 RL 后的 OOD 测试成功率: {success_count}/{num_rollouts} = {success_count/num_rollouts*100}%")

if __name__ == '__main__':
    evaluate_rl_success_rate()