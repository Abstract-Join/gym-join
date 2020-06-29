from queue import PriorityQueue
from heapq import heappush, heappop
from gym_join.envs.db_models import Table, OuterRelationPage
from random import randint, Random
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
PAGE_SIZE = "page_size"
R_SEED = "r_random_seed"
S_SEED = "s_random_seed"

class State:

    def __init__(self, page:OuterRelationPage, page_no, r_disc=20):
        self.page = page
        self.blocks_read = 0
        self.tuples_joined = 0
        self.page_no = page_no
        self.tuples_tried = 0
        self.curr_tuples_tried = 0
        # self.curr_blocks_tried = 0
        self.state_id = 0
        self.r_disc = r_disc

    def get_observation(self):
        values = []
        # values.append(self.blocks_read)
        values.append(self.tuples_joined % self.r_disc)
        # values.append(self.page_no)
        # values.append(self.curr_blocks_tried)
        values.append(self.state_id)

        return np.array(values)


def get_observation_space():
    observation = np.array([
    # np.finfo(np.float).max,
    np.iinfo(np.int16).max,
    # np.finfo(np.float).max,
    np.iinfo(np.int16).max])
    return spaces.Box(-observation, observation)

class JoinEnv(gym.Env):
    def __init__(self):
        #TODO Limit heapsize
        self._max_heap = []
        self._current_state = None

        self.results = set()
        self.action_space = spaces.Discrete(2)
        self.observation_space = get_observation_space()

    def set_config(self, config):
        env_config = config["env"]
        self.config = env_config
        self.k_array = env_config["k_join_array"]
        self._r_table = Table(env_config[OUTER_TABLE_PATH], env_config[PAGE_SIZE], env_config[R_SEED], True)
        self._s_path = env_config[INNER_TABLE_PATH]
        self._s_table = Table(env_config[INNER_TABLE_PATH], env_config[PAGE_SIZE], env_config[S_SEED], False)
        # self.min_heap_size = args.heap_size
        self.page_size = env_config[PAGE_SIZE]
        self.s_seed = env_config[S_SEED]
        self.k = env_config["k"]
    
    def reset(self):
        self._r_table.reset_table()
        self._s_table.reset_table()

        page = self._r_table.next_page()
        state = State(page, self._r_table.page_no - 1, self.config["reward_discretizer"])
        self._current_state = state

        return self._current_state.get_observation()
    
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
        action = self.__redirect_action(action)
        if action == STAY:
            # if self._current_state.curr_tuples_tried >= self._s_table.size:
            #     self.__action_forward()

            reward = self.__action_stay(False)

            return self._current_state.get_observation(), reward, self.__is_done()

        elif action == FORWARD:
            self.__action_forward()
            if self._current_state.page == None:
                return self._current_state.get_observation(), 0, True

            reward = self.__action_stay(True)
            return self._current_state.get_observation(), reward, self.__is_done()
        

        elif action == JOIN_ALL:
            #TODO Do we need to return the current state as well, for in cases where the state's reward changes (JUMP, JOIN_ALL), and then you move forward
            success = self.__join_all(self._current_state.page)

            #Go forward to the next 
            self.__action_forward()
            if self._current_state.page == None:
                return self._current_state.get_observation(), 0, True
            return self._current_state.get_observation(), success, self.__is_done()

        # elif action == JUMP:
        #     if len(self._max_heap) == 0:
        #         return self._current_state.get_observation(), 0, self.__is_done()
        #     _, _, state = heappop(self._max_heap)
        #     success = self.__join_all(state.page)
        #     # self.__get_next_state()
        #     return self._current_state.get_observation(), success, self.__is_done()
        #     # return self.current_state, success + self.__s_table.size, self.__is_done()


    
    def __is_done(self):
        if len(self.results) < self.k:
            return False
        return True
    
    def __redirect_action(self, action):
        if self._current_state.state_id == len(self.k_array) - 1:
            if action == 0:
                action = 2
        return action

    def __action_stay(self, is_initial_state):
        if is_initial_state:
            return self.__action_k_stay(1)

        self._current_state.state_id += 1

        success = self.__action_k_stay(self.k_array[self._current_state.state_id - 1])

        return success
    
    # perform k join 
    def __action_k_stay(self, k):
        s = 0
        for _ in range(k):
            next_s = self._s_table.next_page()
            
            success = self.__join(self._current_state.page, next_s)

            self._current_state.tuples_joined += success
            # self._current_state.tuples_tried += len(next_s)
            # self._current_state.curr_tuples_tried += len(next_s)
            # self._current_state.curr_blocks_tried += 1
            
            s += success
        return s
    
    def __action_forward(self):
   
        page = self._r_table.next_page()
        if page == None:
            self._current_state.page = None
            return
        self._current_state.page = page
        self._current_state.state_id = 0
        # self._current_state.page_no = self._r_table.page_no - 1
        # self._current_state.blocks_read += 1
        # self._current_state.curr_blocks_tried = 0
        # self._current_state.curr_tuples_tried = 0
        self._current_state.tuples_joined = 0

    def __join(self, outer_page, inner_relation_tuples):
        count = 0
        for inner_tuple in inner_relation_tuples:
            if inner_tuple.id1 in outer_page.id1_set:
                self.results.add(inner_tuple.id1 + "-" + inner_tuple.id2)
                count += 1
                if len(self.results) >= self.k:
                    return count
        return count


    def __join_all(self, outer_page):
        s = Table(None, self.page_size, self.s_seed, False, False, self._s_table.table)

        s_page  = s.next_page()
        count = 0
        while s_page:

            for inner_tuple in s_page:

                if inner_tuple.id1 in outer_page.id1_set:
                    self.results.add(inner_tuple.id1 + "-" + inner_tuple.id2)
                    count += 1
                    
                    if len(self.results) >= self.k:
                        return count
            s_page  = s.next_page()
        return count


