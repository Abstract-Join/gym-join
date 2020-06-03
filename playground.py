
from gym_join.envs.join_env import STAY, FORWARD, JUMP, JOIN_ALL
import math
from datetime import datetime
import yaml
import gym
from gym_join.envs.db_models import Table
# r = Table("customer_cleaned.tbl", 16, True)
# s = Table("order.tbl", 16, False, False)

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



def nested_loop_join(config):
    env = gym.make('gym_join:join-v0')
    env.set_config(config)

    done = False
    env.reset()
    while not done:
        
        state, reward, done = env.step(JOIN_ALL)

    # print(env.results)


if __name__ == "__main__":
    with open('config.yml') as f:
        config = yaml.safe_load(f)
    env_config = config["env"]
    # start = datetime.now()
    # mruns(config)
    # print(datetime.now() - start)

    # _r_table = Table(env_config["outer_table_path"], env_config["page_size"], env_config["random_seed"], True)

    # while True:
    #     page = _r_table.next_page()
    #     if page == None:
    #         break
    #     # print(len(page.customer_id_set))

    print("Nested")
    start = datetime.now()

    nested_loop_join(config)
    print(datetime.now() - start)
