from lerobot.teleoperators.teleoperator import Teleoperator
from .zima_ps3_teleop_config import ZimaPS3TeleopConfig
from .gamepad_utils import PS3GamepadController
from typing import Dict, Any


class ZimaPS3Teleop(Teleoperator):
    config_class = ZimaPS3TeleopConfig
    name = "zima_ps3_teleop"

    def __init__(self, config: ZimaPS3TeleopConfig):
        super().__init__(config)
        self.config = config
        self.gamepad: PS3GamepadController = None

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "tracks.left_vel": float,
            "tracks.right_vel": float,
            "arm.pos.delta_x": float,
            "arm.pos.delta_y": float,
            "arm.pos.delta_z": float,
            "arm.rot.delta_x": float,
            "arm.rot.delta_y": float,
            "arm.rot.delta_z": float,
            "arm.gripper.delta": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.gamepad is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        self.gamepad = PS3GamepadController(deadzone=self.config.deadzone)
        self.gamepad.start()
        print(f"PS3 controller connected for Zima teleoperation")

    def disconnect(self) -> None:
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("PS3 controller is not connected")

        state = self.gamepad.get_state()

        linear = 0.0
        angular = 0.0

        if state['dpad_up']:
            linear = 1.0
        elif state['dpad_down']:
            linear = -1.0

        if state['dpad_left']:
            angular = 1.0
        elif state['dpad_right']:
            angular = -1.0

        left_speed = linear - angular
        right_speed = linear + angular

        max_value = max(abs(left_speed), abs(right_speed), 1.0)
        left_vel = (left_speed / max_value) * self.config.max_speed
        right_vel = (right_speed / max_value) * self.config.max_speed

        left_x = state['left_stick_x']
        left_y = state['left_stick_y']
        right_x = state['right_stick_x']
        right_y = state['right_stick_y']

        gripper_delta = 0.0
        if state['l2'] > 0.5:
            gripper_delta -= 1.0
        if state['r2'] > 0.5:
            gripper_delta += 1.0

        arm_delta_x = -left_y * self.config.max_arm_delta
        arm_delta_y = -left_x * self.config.max_arm_delta
        arm_delta_z = -right_y * self.config.max_arm_delta

        arm_rot_delta_x = state['l1'] * -self.config.max_rotation_delta + state['r1'] * self.config.max_rotation_delta
        arm_rot_delta_y = 0.0
        arm_rot_delta_z = 0.0

        return {
            "tracks.left_vel": float(left_vel),
            "tracks.right_vel": float(right_vel),
            "arm.pos.delta_x": float(arm_delta_x),
            "arm.pos.delta_y": float(arm_delta_y),
            "arm.pos.delta_z": float(arm_delta_z),
            "arm.rot.delta_x": float(arm_rot_delta_x),
            "arm.rot.delta_y": float(arm_rot_delta_y),
            "arm.rot.delta_z": float(arm_rot_delta_z),
            "arm.gripper.delta": float(gripper_delta),
        }

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        pass
