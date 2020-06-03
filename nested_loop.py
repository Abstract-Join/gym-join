from datetime import datetime
import yaml
from gym_join.envs.db_models import Table

OUTER_TABLE_PATH = "outer_table_path"
INNER_TABLE_PATH = "inner_table_path"
PAGE_SIZE = "page_size"


def nested_loop_join(config):
    env_config = config["env"]
    r_table = Table(env_config[OUTER_TABLE_PATH], env_config[PAGE_SIZE], env_config["random_seed"], True)
    k = env_config["k"]
    result_set = set()
    while True:

        r_page = r_table.next_page()
        if r_page == None:
            break
        s_table = Table(env_config[INNER_TABLE_PATH], env_config[PAGE_SIZE], env_config["random_seed"], False, False)

        while True:
            s_page = s_table.next_page()
            if s_page == None:
                break
            for s_tuple in s_page:

                if s_tuple.customer_id in r_page.customer_id_set:
                    result_set.add(s_tuple.customer_id + "-" + s_tuple.order_id)

                    if len(result_set) >= k:
                        return result_set
        print(len(result_set))


if __name__ == "__main__":
    with open('config.yml') as f:
        config = yaml.safe_load(f)
    env_config = config["env"]

    print("Block Nested Loop Join")
    start = datetime.now()
    nested_loop_join(config)
    print("Time taken : " + str(datetime.now() - start))