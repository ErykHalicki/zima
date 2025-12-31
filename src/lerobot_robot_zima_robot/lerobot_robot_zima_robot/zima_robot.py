from lerobot.robots import Robot
from .zima_robot_config import ZimaRobotConfig

class ZimaRobot(Robot):
    config_class = ZimaRobotConfig
    name = "zima_robot"

    def __init__(self, config: ZimaRobotConfig):
        super().__init__(config)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "track_left.pos": float, #TODO rename to be better
            "track_right.pos": float,
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            "wrist_cam": (480, 640, 3), #HWC, opencv format
            "front_cam": (480, 640, 3),
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

