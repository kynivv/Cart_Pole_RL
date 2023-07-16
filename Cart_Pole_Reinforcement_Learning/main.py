import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import warnings
warnings.filterwarnings('ignore')


# Creating the Environment
environment_name = 'CartPole-v0'
env = gym.make(environment_name)

episodes = 5

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        step_result = env.step(action)
        n_state, reward, done, info = step_result[:4]
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    
env.close()


# Training Algorithm
log_path = 'output/'
training_log_path = os.path.join(log_path, 'PPO_3')
#!tensorboard --logdir={training_log_path}

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

stop_callback = StopTrainingOnRewardThreshold(reward_threshold= 190, verbose= 1)
eval_callback = EvalCallback(env,
                             callback_on_new_best= stop_callback,
                             eval_freq= 10000,
                             best_model_save_path= save_path,
                             verbose= 1)

model = PPO('MlpPolicy', env, verbose= 1, tensorboard_log= log_path)

model.learn(total_timesteps= 2000, callback= eval_callback)

model_path = os.path.join('Training', 'Saved Models', 'best_model')

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

ppo_path = os.path.join('Training', 'Saved Models', 'best_model')

model.save(ppo_path)