from environment import LoRaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

env = LoRaEnv()
check_env(env)
env = make_vec_env(lambda: env, n_envs=1)

model = PPO('MlpPolicy', env, verbose=0, gamma=0.9, learning_rate=0.001, batch_size=512)
model.learn(total_timesteps=6000000)

version = 0

# Save the trained agent
model.save(f"lora_rl_ppo_v{version}")