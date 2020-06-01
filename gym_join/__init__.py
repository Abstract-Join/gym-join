from gym.envs.registration import register

register(
    id='join-v0',
    entry_point='gym_join.envs:JoinEnv'
)