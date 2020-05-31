
from db_models import Table, join
from env import Env, STAY, FORWARD, JUMP, JOIN_ALL
import math
from datetime import datetime

# r = Table("customer_cleaned.tbl", 16, True)
# s = Table("order.tbl", 16, False, False)

def mruns():

    env = Env(None)
    m = int(math.sqrt(env.r_table.size / 32))
    state = env.reset()
    done = False

    i = 0
    last_join = 0
    while not done:
        
        if i == 0:
            next_state, reward, done = env.step(STAY)
            i += 1
            continue

        if reward >= m:
            next_state, reward, done = env.step(JOIN_ALL)
            reward = 0
            last_join = i
        elif i - last_join >= m:
            next_state, reward, done = env.step(JUMP)
            reward = 0
            last_join = i

        else:
            next_state, reward, done = env.step(FORWARD)
        
        i += 1
    
    # print(env.results)

            
            


        

        # join(s_result, results)

    # print(results)


def nested_loop_join():
    env = Env(None)

    r_page = env.r_table.next_page()
    done = False
    env.reset()
    while not done:
        
        state, reward, done = env.step(JOIN_ALL)

    # print(env.results)


if __name__ == "__main__":
    start = datetime.now()
    mruns()
    print(datetime.now() - start)

    print("Nested")
    start = datetime.now()

    nested_loop_join()
    print(datetime.now() - start)
