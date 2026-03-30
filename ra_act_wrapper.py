import gym
import numpy as np
import torch
from einops import rearrange

# 假设原有的常量和预处理逻辑
from constants import DT
from sim_env import BOX_POSE
from utils import sample_box_pose, sample_box_pose_eval, sample_box_pose_for_excavator, sample_insertion_pose # robot functions

class RA_ACT_Wrapper(gym.Env):
    """
    将 dm_control 环境与冻结的 ACT 模型封装为标准的 Gym 环境
    """
    def __init__(self, env, act_policy, stats, config, epsilon=0.05, lambda_penalty=0.1):
        super().__init__()
        self.env = env
        self.act_policy = act_policy
        self.stats = stats       # 包含 qpos_mean, qpos_std, action_mean, action_std
        self.config = config
        
        # 提取配置参数
        self.camera_names = config['camera_names']
        self.query_freq = config['policy_config']['num_queries']
        self.state_dim = config['state_dim']
        self.temporal_agg = config.get('temporal_agg', False)
        
        # RL 控制参数
        self.epsilon = epsilon            # 残差物理上限
        self.lambda_penalty = lambda_penalty  # 动作平滑惩罚系数
        
        # 冻结模型
        self.act_policy.eval()
        for param in self.act_policy.parameters():
            param.requires_grad = False
            
        self.step_count = 0
        self.current_chunk = None
        self._last_ts = None
        
        # 定义动作空间 (RL 输出 [-1, 1])
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)
        
        BOX_POSE[0] = sample_box_pose_eval()

        # 定义观测空间 (需要实际 reset 一次来获取确切维度)
        self._last_ts = self.env.reset()
        a_nom_dummy = np.zeros(self.state_dim)
        dummy_obs = self._get_rl_obs(self._last_ts.observation, a_nom_dummy)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32)

    def reset(self):
        self.step_count = 0
        self._last_ts = self.env.reset()
        
        # FIXME: 之后在第三步时，这里要替换为强制的 OOD Reset (比如 sample_box_pose_eval_ring)
        
        # 获取初始 ACT 动作块
        self.current_chunk = self._query_act(self._last_ts.observation)
        a_nom = self.current_chunk[0]
        
        return self._get_rl_obs(self._last_ts.observation, a_nom)

    def step(self, residual_action):
        # 1. 检查是否需要更新 Chunk
        if self.step_count % self.query_freq == 0 and self.step_count > 0:
            self.current_chunk = self._query_act(self._last_ts.observation)
            
        # 2. 提取当前步骤的名义动作
        chunk_index = self.step_count % self.query_freq
        a_nom = self.current_chunk[chunk_index]
        
        # 3. 动作融合 (残差映射到 [-epsilon, epsilon])
        delta_a = np.tanh(residual_action) * self.epsilon # 这里目前给到的是0,没有影响act的输出
        a_env = a_nom + delta_a # 拿到最终动作 = ACT 输出 + RL 给的残差，当前 residual_action 是全 0 的，所以 a_env 就完全等于 a_nom
        
        # 4. 执行物理仿真
        self._last_ts = self.env.step(a_env)
        
        # 5. 获取下一步给 RL 的观测
        next_chunk_idx = (self.step_count + 1) % self.query_freq
        next_a_nom = self.current_chunk[next_chunk_idx] if next_chunk_idx != 0 else a_nom
        rl_obs = self._get_rl_obs(self._last_ts.observation, next_a_nom)
        
        # 6. 计算 Reward (这里暂时只透传原环境 reward，供 Dummy Test 使用)
        base_reward = self._last_ts.reward if self._last_ts.reward is not None else 0.0
        penalty = self.lambda_penalty * np.linalg.norm(delta_a)**2
        rl_reward = base_reward - penalty
        
        done = self._last_ts.last()
        self.step_count += 1
        
        info = {'base_reward': base_reward, 'penalty': penalty}
        return rl_obs, rl_reward, done, info

    @torch.inference_mode()
    def _query_act(self, obs):
        """完全复刻模仿学习的评估预处理逻辑"""
        # 图像处理
        curr_images = []
        for cam_name in self.camera_names:
            curr_image = rearrange(obs['images'][cam_name], 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        
        # qpos 处理
        qpos_numpy = np.array(obs['qpos'])
        qpos_normalized = (qpos_numpy - self.stats['qpos_mean']) / self.stats['qpos_std']
        qpos = torch.from_numpy(qpos_normalized).float().cuda().unsqueeze(0)
        
        # 推理
        all_actions = self.act_policy(qpos, curr_image)
        all_actions = all_actions.squeeze(0).cpu().numpy()
        
        # 反归一化为物理空间控制指令
        post_processed_actions = all_actions * self.stats['action_std'] + self.stats['action_mean']
        return post_processed_actions

    def _get_rl_obs(self, obs, a_nom):
        """拼接成稠密一维向量喂给 RL (MLP 网络)"""
        qpos = np.array(obs['qpos'])
        qvel = np.array(obs['qvel'])
        env_state = np.array(obs['env_state']) # 这里面包含 Box Pose 
        
        return np.concatenate([qpos, qvel, env_state, a_nom], axis=0).astype(np.float32)