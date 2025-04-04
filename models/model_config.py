ppo_config = {
"PPO_SCENARIO1_CONFIG":
{
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
    "total_timesteps": 50000,
    "n_envs": 1,
},

"PPO_SCENARIO2_CONFIG":
{
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
    "total_timesteps": 10000,
    "n_envs": 1,
}
}

test_episodes = 10
