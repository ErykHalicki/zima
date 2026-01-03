from lerobot.robots import Robot
from .zima_robot_config import ZimaRobotConfig
import socket
import json
import numpy as np
import cv2
import base64
from typing import Dict, Any

class ZimaRobot(Robot):
    config_class = ZimaRobotConfig
    name = "zima_robot"

    def __init__(self, config: ZimaRobotConfig):
        super().__init__(config)
        self.socket = None
        self._connected = False

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
    def is_connected(self) -> bool:
        return self._connected and self.socket is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.config.address, int(self.config.port)))
            self._connected = True
            print(f"Connected to Zima robot at {self.config.address}:{self.config.port}")
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to robot: {e}")

    def disconnect(self) -> None:
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                print(f"Error closing socket: {e}")
            finally:
                self.socket = None
                self._connected = False

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("Robot is not connected")

        request = {"command": "get_observation"}
        response = self._send_tcp_request(request)

        if response.get("status") != "success":
            raise RuntimeError(f"Failed to get observation: {response.get('message', 'Unknown error')}")

        obs_data = response.get("data", {})

        obs = {}

        for key in self._joint_ft.keys():
            obs[key] = obs_data.get(key, 0.0)

        for cam_key in self._cameras_ft.keys():
            cam_data_b64 = obs_data.get(cam_key)
            if cam_data_b64 is not None:
                jpeg_bytes = base64.b64decode(cam_data_b64)
                np_arr = np.frombuffer(jpeg_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if image is not None:
                    obs[cam_key] = image
                else:
                    obs[cam_key] = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                obs[cam_key] = np.zeros((480, 640, 3), dtype=np.uint8)

        return obs

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("Robot is not connected")

        action_data = {}
        for key in self.action_features.keys():
            action_data[key] = float(action.get(key, 0.0))

        request = {
            "command": "send_action",
            "data": action_data
        }

        response = self._send_tcp_request(request)

        if response.get("status") != "success":
            raise RuntimeError(f"Failed to send action: {response.get('message', 'Unknown error')}")

        return action

    def _send_tcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            request_str = json.dumps(request)
            self.socket.sendall(request_str.encode('utf-8'))

            response_data = b""
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                try:
                    response = json.loads(response_data.decode('utf-8'))
                    return response
                except json.JSONDecodeError:
                    continue

            raise RuntimeError("Connection closed by server")

        except Exception as e:
            self._connected = False
            raise RuntimeError(f"TCP communication error: {e}")
