from gym.envs.registration import register

register(
    id='bandits-v0',
    entry_point='gym_bandits.envs:Bandits',
)