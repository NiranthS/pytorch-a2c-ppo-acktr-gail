import pytest
import numpy as np

from stable_baselines3 import PPO
# from stable_baselines3.ddpg import NormalActionNoise
from stable_baselines3.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, seed=0, lam=0.5,
                           optim_batchsize=16, optim_stepsize=1e-3).learn(total_timesteps=15000),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e, seed=0,
                           learning_rate=1.5e-3, lam=0.8).learn(total_timesteps=20000)

}


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ['a2c', 'acer', 'acktr', 'dqn', 'ppo1', 'ppo2', 'trpo'])
def test_identity(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    :param model_name: (str) Name of the RL model
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    model = LEARN_FUNC_DICT[model_name](env)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)

    obs = env.reset()
    assert model.action_probability(obs).shape == (1, 10), "Error: action_probability not returning correct shape"
    action = env.action_space.sample()
    action_prob = model.action_probability(obs, actions=action)
    assert np.prod(action_prob.shape) == 1, "Error: not scalar probability"
    action_logprob = model.action_probability(obs, actions=action, logp=True)
    assert np.allclose(action_prob, np.exp(action_logprob)), (action_prob, action_logprob)

    # Free memory
    del model, env


# @pytest.mark.slow
# @pytest.mark.parametrize("model_class", [DDPG, TD3, SAC])
def test_identity_continuous():
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

    if model_class in [DDPG, TD3]:
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    else:
        action_noise = None

    model = model_class("MlpPolicy", env, gamma=0.1, seed=0,
                         action_noise=action_noise, buffer_size=int(1e6))
    model.learn(total_timesteps=20000)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)
    # Free memory
    del model, env

test_identity_continuous()