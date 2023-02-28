import logging
import coloredlogs
import aicrowd_gym
import minerl
from config import EVAL_EPISODES, EVAL_MAX_STEPS
from agents import Agent, VPTAgent

coloredlogs.install(logging.DEBUG)

MINERL_GYM_ENV = 'MineRLObtainDiamondShovel-v0'


def main():
    # NOTE: It is important that you use "aicrowd_gym" instead of regular "gym"!
    #       Otherwise, your submission will fail.
    env = aicrowd_gym.make(MINERL_GYM_ENV)

    # Create agent
    agent = VPTAgent(env)

    # Load your model here
    # NOTE: The trained parameters must be inside "train" directory!
    # model = None

    for i in range(EVAL_EPISODES):
        obs = env.reset()
        done = False
        for step_counter in range(EVAL_MAX_STEPS):

            # get next action from agent
            act = agent.get_action(obs)

            obs, reward, done, info = env.step(act)

            if done:
                break
        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()


if __name__ == "__main__":
    main()
