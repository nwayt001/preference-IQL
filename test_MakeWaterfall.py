import logging
import coloredlogs
import aicrowd_gym
import minerl
from config import EVAL_EPISODES, EVAL_MAX_STEPS
from agents import Agent, VPTAgent, IQLearnAgent

coloredlogs.install(logging.DEBUG)

MINERL_GYM_ENV = 'MineRLBasaltMakeWaterfall-v0'
WEIGHTS_FILE = 'sac_iq_MineRLBasaltMakeWaterfall-v0.weights'

def main():
    # NOTE: It is important that you use "aicrowd_gym" instead of regular "gym"!
    #       Otherwise, your submission will fail.
    env = aicrowd_gym.make(MINERL_GYM_ENV)

    # Create agent
    agent = IQLearnAgent(env, weights=WEIGHTS_FILE)

    for i in range(EVAL_EPISODES):
        obs = env.reset()
        agent.reset_agent()
        done = False
        episode_end = False
        end_cntr = 0
        for step_counter in range(EVAL_MAX_STEPS):

            # get next action from agent
            act = agent.get_action(obs)
            act["drop"] = 0
            act["inventory"] = 0
            act["sneak"] = 0
            if act["use"] == 1:
                print("use pressed")
                episode_end = True

            obs, reward, done, info = env.step(act)

            if episode_end == True:
                end_cntr += 1
            
            if episode_end == True and end_cntr > 100:
                act["ESC"] = 1
                obs, reward, done, info = env.step(act)
                done = True

            if done:
                break
        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()
