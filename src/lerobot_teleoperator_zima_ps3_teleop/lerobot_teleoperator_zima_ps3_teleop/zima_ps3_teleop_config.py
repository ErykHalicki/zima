from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("zima_ps3_teleop")
@dataclass
class ZimaPS3TeleopConfig(TeleoperatorConfig):
    deadzone: float = 0.15
    max_speed: float = 1.0
    max_arm_delta: float = 1.0
    max_rotation_delta: float = 1.0
