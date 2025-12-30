from lerobot.robots import Robot
from .zima_robot_config import ZimaRobotConfig 

class ZimaRobot(Robot):
    config_class = ZimaRobotConfig
    name = "zima_robot"

    def __init__(self, config: ZimaRobotConfig):
        super().__init__(config)
        
