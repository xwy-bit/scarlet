import sys
sys.path.append(".")

from utils.env import TaskLess_Env
import habitat
from utils.logger import logger


def  run():

    config = habitat.get_config(
        config_path="configs/pointgoal_test.yaml"
    )    
    env = TaskLess_Env(config,logger=logger)
    logger.info("eposide_dir : {}".format(env.eposide_dir))
    for ii in range(10):
        env.load_new_scene()

if __name__ == "__main__":
    run()
