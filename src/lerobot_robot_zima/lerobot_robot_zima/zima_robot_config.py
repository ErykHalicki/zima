from dataclasses import dataclass
from lerobot.robots import RobotConfig

@RobotConfig.register_subclass("zima")
@dataclass
class ZimaRobotConfig(RobotConfig):
    port: str
    address: str
