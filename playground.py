
from gym_join.envs.join_env import STAY, FORWARD, JUMP, JOIN_ALL
import math
from datetime import datetime
import yaml
import gym
# r = Table("customer_cleaned.tbl", 16, True)
# s = Table("order.tbl", 16, False, False)

def mruns(config):

    env = gym.make('gym_join:join-v0')
    env.set_config(config)
    m = int(math.sqrt(env._r_table.size / 32))
    state = env.reset()
    done = False
    env.action_space
    i = 0
    last_join = 0
    while not done:
        next_state, reward, done = env.step(STAY)
        print(next_state)
        # if i == 0:
        #     next_state, reward, done = env.step(STAY)
        #     print(next_state)
        #     i += 1
        #     continue

        # if reward >= m:
        #     next_state, reward, done = env.step(JOIN_ALL)
        #     print(next_state)

        #     reward = 0
        #     last_join = i
        # elif i - last_join >= m:
        #     next_state, reward, done = env.step(JUMP)
        #     print(next_state)

        #     reward = 0
        #     last_join = i

        # else:
        #     next_state, reward, done = env.step(FORWARD)
        #     print(next_state)

        # i += 1
    
    # print(env.results)

            
            


        

        # join(s_result, results)

    # print(results)


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
    start = datetime.now()
    mruns(config)
    print(datetime.now() - start)

    # print("Nested")
    # start = datetime.now()

    # nested_loop_join(config)
    # print(datetime.now() - start)
