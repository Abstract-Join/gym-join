from queue import PriorityQueue
from heapq import heappush, heappop
from db_models import Table, OuterRelationPage
from random import randint

STAY = 0
FORWARD = 1
JOIN_ALL = 2
JUMP = 3
r_path = "data/customer_cleaned.tbl"
s_path = "data/order.tbl"

# >>> random.Random(4).shuffle(x)

class State:

    def __init__(self, page:OuterRelationPage, page_no, reward = 0, page_size = 32):
        self.page = page
        self.page_no = page_no,
        self.reward = reward
        #Included page size in case the last page fewer than default page size
        # self.page_size = page_size
        self.pages_joined = 0

class Env(object):
    def __init__(self, args):
        # self.R_rand_seed = args.R_rand_seed
        # self.S_rand_seed = args.S_rand_seed

        #TODO Limit heapsize
        self.max_heap = []
        self.current_state = None
        self.r_table = Table(r_path, 16, True)
        self.s_table = Table(s_path, 16, False)
        # self.min_heap_size = args.heap_size
        self.k = 10000
        self.results = set()
        self.actions = [STAY, FORWARD, JUMP, JOIN_ALL]
    
    def reset(self):
        self.r_table.reset_table()
        self.s_table.reset_table()
        return self.__get_next_state()


    def step(self, action):
        """
        :param action:
        :return:
            :next state
            :reward
            :done
        """

        if action == STAY:
            self.__action_stay()
            return self.current_state, self.current_state.reward, self.__is_done()

        elif action == FORWARD:
            # print(self.current_state.reward)
            try:
                heappush(self.max_heap, (-self.current_state.reward, randint(0, 10000), self.current_state))
            except:
                print("Collision")
            ##TODO Handle case when outer tuple is completed but we havent found k results. Would have to limit the other actions and allow only JUMP
            # if next_r == None:
                #When outer tuple is completed but k is still not done
            
            ##TODO Need to clarify if when after doing a forward, we need to do a stay as well
            self.__get_next_state()
            self.__action_stay()
            return self.current_state, self.current_state.reward, self.__is_done()

        elif action == JUMP:
            _, _, state = heappop(self.max_heap)
            success = self.join_all(state.page)
            self.__get_next_state()
            return self.current_state, success, self.__is_done()
            # return self.current_state, success + self.s_table.size, self.__is_done()

        elif action == JOIN_ALL:
            #TODO Should the reward for when you do a jump be the same like success + tuples read(Should it even be tuples read or blocks read)
            #TODO Do we need to return the current state as well, for in cases where the state's reward changes (JUMP, JOIN_ALL), and then you move forward
            success = self.join_all(self.current_state.page)
            self.__get_next_state()
            # return self.current_state, success + self.s_table.size, self.__is_done()
            return self.current_state, success, self.__is_done()


    def action_space(self):
        return len(self.actions)
    
    def __is_done(self):
        if len(self.results) < self.k:
            return False
        return True
    
    def __action_stay(self):
        next_s = self.s_table.next_page()
        success = self.join(self.current_state.page, next_s)
        self.current_state.reward += success - len(next_s)
        # self.current_state.reward += success

        self.current_state.pages_joined += 1

    
    def __get_next_state(self):
        page = self.r_table.next_page()
        state = State(page, self.r_table.page - 1)
        self.current_state = state
        return state


    def join(self, outer_page, inner_relation_tuples):
        count = 0
        for inner_tuple in inner_relation_tuples:

            if inner_tuple.customer_id in outer_page.customer_id_set:
                self.results.add(inner_tuple.customer_id + "-" + inner_tuple.order_id)
                count += 1
                if len(self.results) > self.k:
                    return count
        return count


    def join_all(self, outer_page):
        s = Table(s_path, 16, False, False)

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
