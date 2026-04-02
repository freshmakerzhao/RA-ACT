import os
import torch
import pickle
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# 导入你自己的代码
from constants import load_config, get_training_config, get_equipment_model, get_sim_task_config
from sim_env import make_sim_env
from policy import ACTPolicy
from ra_act_wrapper import RA_ACT_Wrapper

def train_residual_sac():
    # ================= 1. 初始化和加载预备环境 =================
    config_path = "configs/fairino5_single_act_base/03_eval.yaml" 
    yaml_config = load_config(config_path)
    training_config = get_training_config(config_path)
    
    task_name = yaml_config.get('task', {}).get('name', 'sim_lifting_cube_scripted')
    equipment_model = get_equipment_model(config_path)
    ckpt_dir = training_config.get('ckpt_dir', './ckpts/fairino5_single_act_base')
    task_config = get_sim_task_config(task_name, config_path)
    camera_names = task_config['camera_names']
    
    # 加载 stats
    with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)

    # 加载预训练的 ACT
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

    # ================= 2. 包装强化学习环境 =================
    base_env = make_sim_env(task_name, equipment_model)
    wrapper_config = {
        'camera_names': camera_names,
        'policy_config': policy_config,
        'state_dim': 7
    }
    
    # Wrapper
    # epsilon 控制残差的最大修正幅度，如果是弧度制，0.05~0.1 左右比较安全
    rl_env = RA_ACT_Wrapper(base_env, act_policy, stats, wrapper_config, epsilon=0.08)

    # ================= 3. 定义 SAC 算法模型 =================
    log_dir = "./sac_ra_act_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 采用 MLP 网络 (因为我们给 RL 的状态是拼接好的一维向量)
    model = SAC(
        "MlpPolicy", 
        rl_env, 
        learning_rate=3e-4, 
        buffer_size=100000, 
        batch_size=256,
        ent_coef='auto', # 自动调节温度参数以鼓励探索
        gamma=0.99,      # 折扣因子
        tensorboard_log=log_dir,
        verbose=1,
        device="cuda"
    )

    # 回调函数：每 10000 步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=log_dir, 
        name_prefix="ra_act_sac_model"
    )

    print("🚀 开始在 OOD 区域训练残差网络！")
    # 开始训练，你可以先跑 100,000 步试试水，大概十几分钟就能看到收敛趋势
    model.learn(total_timesteps=100000, callback=checkpoint_callback)
    
    # 保存最终模型
    model.save(os.path.join(log_dir, "ra_act_sac_final"))
    print("✅ 训练完成，模型已保存！")

if __name__ == '__main__':
    train_residual_sac()