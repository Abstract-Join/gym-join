from gym_join.envs.join_env import STAY, FORWARD, JUMP, JOIN_ALL
import math
from datetime import datetime
import yaml
import gym
from gym_join.envs.db_models import Table

OUTER_TABLE_PATH = "outer_table_path"
INNER_TABLE_PATH = "inner_table_path"
PAGE_SIZE = "page_size"

def mruns(config):

    env = gym.make('gym_join:join-v0')
    env.set_config(config)
    # m = int(math.sqrt(env._r_table.size / 32))
    state = env.reset()
    done = False
    env.action_space
    i = 0
    last_join = 0
    while not done:
        next_state, reward, done = env.step(STAY)
        print(len(env.results))

        print(next_state)
        # break
    
    print(len(env.results))


if __name__ == "__main__":
    with open('config.yml') as f:
        config = yaml.safe_load(f)
    env_config = config["env"]
    start = datetime.now()
    mruns(config)
    print(datetime.now() - start)
