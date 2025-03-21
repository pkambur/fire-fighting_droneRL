PPO_DEFAULT_CONFIG = {
    "policy": "MlpPolicy",
    "verbose": 1,
    "learning_rate": 0.0001,
    "n_steps": 4096,
    "batch_size": 256,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": 0.2,
    "ent_coef": 0.01,
    "total_timesteps": 100000,
}

test_episodes = 10
