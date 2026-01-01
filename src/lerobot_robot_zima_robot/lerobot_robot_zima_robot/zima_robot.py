from lerobot.robots import Robot
from .zima_robot_config import ZimaRobotConfig

class ZimaRobot(Robot):
    config_class = ZimaRobotConfig
    name = "zima_robot"

    def __init__(self, config: ZimaRobotConfig):
        super().__init__(config)

    @property
    def _action_ft(self) -> dict[str, type]:
        return {
            "tracks.left_vel": float, 
            "tracks.right_vel": float,
            "arm.pos.delta_x" : float,
            "arm.pos.delta_y" : float,
            "arm.pos.delta_z" : float,
            "arm.rot.delta_x" : float,
            "arm.rot.delta_y" : float,
            "arm.rot.delta_z" : float,
            "arm.gripper.delta" : float,
        }

    @property
    def _joint_ft(self) -> dict[str, type]:
        return {
            "tracks.left_vel": float, 
            "tracks.right_vel": float,
            "joint1.pos": float, 
            "joint2.pos": float, 
            "joint3.pos": float, 
            "joint4.pos": float, 
            "joint5.pos": float, 
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            "wrist_cam": (480, 640, 3), #HWC, opencv format
            "front_cam": (480, 640, 3),
        }

    @property
    def observation_features(self) -> dict:
        return {**self._joint_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._action_ft

