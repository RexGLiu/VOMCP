from gymnasium.envs.registration import register

register(
    id='AdhocReasoningEnv-v1',
    entry_point='src.envs:AdhocReasoningEnv',
)

register(
    id='LaserTagEnv-v1',
    entry_point='src.envs:TagEnv',
)
