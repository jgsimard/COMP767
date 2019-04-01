from gym.envs.registration import register

register(
    id='project-v0',
    entry_point='gym_project.envs:ProjectEnv',
)