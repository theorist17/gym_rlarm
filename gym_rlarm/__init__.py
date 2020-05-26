from gym.envs.registration import register

register(
    id='rlarm-v0',
    entry_point='gym_rlarm.envs:RlarmEnv',
)
register(
    id='rlarm-extrahard-v0',
    entry_point='gym_rlarm.envs:RlarmExtraHardEnv',
)