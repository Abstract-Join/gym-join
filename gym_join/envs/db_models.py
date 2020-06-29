from gym_join.envs.util import load_csv
from random import Random

class OuterRelationPage:

    def __init__(self, id1_set):
        self.id1_set = id1_set


class InnerRelationTuple:

    def __init__(self, id1, id2):
        self.id2 = id2
        self.id1 = id1


class Table:

    def __init__(self, path, page_size, random_seed, isOuter, reset=True, table=None):
        if path == None:
            self.table = table
        else:
            self.table = load_csv(path)
            
        Random(random_seed).shuffle(self.table)

        self.current_index = 0
        self.size = len(self.table)
        self.page_size = page_size
        self.isOuter = isOuter
        self.page_no = 0
        self.reset = reset
    

    def reset_table(self):
        self.page_no = 0
        self.current_index = 0

        
    def next_page(self):
        if self.current_index == -1:
            return None
        start_index, end_index = self.__get_next_indexes()
        page = self.table[start_index:end_index]
        if len(page) > 0:
            self.page_no += 1

        if self.isOuter:
            return self.__set_outer_page(page)
        else:
            return self.__set_inner_tuples(page)
            
    def __set_outer_page(self, page):
        id_set = set()

        for val in page:
            id_set.add(val[0])
        
        if len(id_set) == 0:
            return None
        return OuterRelationPage(id_set)
    
    def __set_inner_tuples(self, page):
        inner_relation_list = []

        for val in page:
            ir = InnerRelationTuple(val[0], val[1])
            inner_relation_list.append(ir)

        if len(inner_relation_list) == 0:
            return None
        return inner_relation_list

    def __get_next_indexes(self):
        
        start_index = self.current_index
        if start_index + self.page_size < self.size:
            end_index = start_index + self.page_size
            self.current_index = end_index
        else:
            end_index = self.size 
            if not self.isOuter and self.reset:
                self.current_index = 0
                #Reset the inner table 
                if start_index == end_index:
                    start_index = 0
                    end_index = start_index + self.page_size + 1
                    self.current_index = end_index
                else:
                    self.current_index = 0
            else:
                self.current_index = -1
        return start_index, end_index






            

