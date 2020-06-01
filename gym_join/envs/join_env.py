from queue import PriorityQueue
from heapq import heappush, heappop
from gym_join.envs.db_models import Table, OuterRelationPage
from random import randint
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

STAY = 0
FORWARD = 1
JOIN_ALL = 2
JUMP = 3

OUTER_TABLE_PATH = "outer_table_path"
INNER_TABLE_PATH = "inner_table_path"

# >>> random.Random(4).shuffle(x)

class State:

    def __init__(self, page:OuterRelationPage, page_no, reward = 0, page_size = 32):
        self.page = page
        self.page_no = page_no
        self.reward = reward
        #Included page size in case the last page fewer than default page size
        # self.page_size = page_size
        self.pages_joined = 0
    
    def get_observation(self):
        values = []
        # values.append(self.page)
        values.append(self.page_no)
        values.append(self.reward)
        values.append(self.pages_joined)
        return np.array(values)


def get_observation_space():
    observation = np.array([
    np.finfo(np.float).max,
    np.finfo(np.float).max,
    np.finfo(np.float).max])
    return spaces.Box(-observation, observation)

class JoinEnv(gym.Env):
    def __init__(self):
        # self.R_rand_seed = args.R_rand_seed
        # self.S_rand_seed = args.S_rand_seed
        #TODO Limit heapsize
        self._max_heap = []
        self._current_state = None

        self.results = set()
        self.action_space = spaces.Discrete(4)
        self.observation_space = get_observation_space()
        # self.actions = [STAY, FORWARD, JUMP, JOIN_ALL]
    
    def set_config(self, config):
        env_config = config["env"]

        self._r_table = Table(env_config[OUTER_TABLE_PATH], 16, True)
        self._s_path = env_config[INNER_TABLE_PATH]
        self._s_table = Table(env_config[INNER_TABLE_PATH], 16, False)
        # self.min_heap_size = args.heap_size
        self.k = env_config["k"]
    
    def reset(self):
        self._r_table.reset_table()
        self._s_table.reset_table()
        return self.__get_next_state()
    
    def get_current_state(self):
        if self._current_state:
            return self._current_state.get_observation()

    def step(self, action):
        """
        :param action:
        :return:
            :next state
            :reward
            :done
        """
        print(action)
        if action == STAY:
            self.__action_stay()
            return self._current_state.get_observation(), self._current_state.reward, self.__is_done()

        elif action == FORWARD:
            # print(self.current_state.reward)
            try:
                heappush(self._max_heap, (-self._current_state.reward, randint(0, 10000), self._current_state))
            except:
                print("Collision")
            ##TODO Handle case when outer tuple is completed but we havent found k results. Would have to limit the other actions and allow only JUMP
            # if next_r == None:
                #When outer tuple is completed but k is still not done
            
            ##TODO Need to clarify if when after doing a forward, we need to do a stay as well
            self.__get_next_state()
            self.__action_stay()
            return self._current_state.get_observation(), self._current_state.reward, self.__is_done()

        elif action == JUMP:
            _, _, state = heappop(self._max_heap)
            success = self.__join_all(state.page)
            self.__get_next_state()
            return self._current_state.get_observation(), success, self.__is_done()
            # return self.current_state, success + self.__s_table.size, self.__is_done()

        elif action == JOIN_ALL:
            #TODO Should the reward for when you do a join_all/jump be the same like success + tuples read(Should it even be tuples read or blocks read)
            #TODO Do we need to return the current state as well, for in cases where the state's reward changes (JUMP, JOIN_ALL), and then you move forward
            success = self.__join_all(self._current_state.page)
            self.__get_next_state()
            # return self.current_state, success + self.__s_table.size, self.__is_done()
            return self._current_state.get_observation(), success, self.__is_done()

    
    def __is_done(self):
        if len(self.results) < self.k:
            return False
        return True
    
    
    def __action_stay(self):
        next_s = self._s_table.next_page()
        success = self.__join(self._current_state.page, next_s)
        # self._current_state.reward += success - len(next_s)
        self._current_state.reward += success

        self._current_state.pages_joined += 1

    
    def __get_next_state(self):
        page = self._r_table.next_page()
        print("Page_no : " + str(self._r_table.page_no - 1))
        state = State(page, self._r_table.page_no - 1)
        self._current_state = state
        return state


    def __join(self, outer_page, inner_relation_tuples):
        count = 0
        for inner_tuple in inner_relation_tuples:

            if inner_tuple.customer_id in outer_page.customer_id_set:
                self.results.add(inner_tuple.customer_id + "-" + inner_tuple.order_id)
                count += 1
                if len(self.results) > self.k:
                    return count
        return count


    def __join_all(self, outer_page):
        s = Table(self._s_path, 16, False, False)

        s_page  = s.next_page()
        count = 0
        while s_page:

            for inner_tuple in s_page:

                if inner_tuple.customer_id in outer_page.customer_id_set:
                    self.results.add(inner_tuple.customer_id + "-" + inner_tuple.order_id)

                    if len(self.results) > self.k:
                        return count
                count += 1
            s_page  = s.next_page()
        return count


