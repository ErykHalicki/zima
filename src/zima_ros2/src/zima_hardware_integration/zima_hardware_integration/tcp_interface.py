#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from zima_msgs.msg import MotorCommand, ArmStateDelta, ServoCommand
from geometry_msgs.msg import Vector3
import socketserver
import threading
import json
import numpy as np
import cv2
import base64
from typing import Dict, Any, Optional


class TCPInterfaceNode(Node):
    def __init__(self):
        super().__init__('tcp_interface_node')

        self.declare_parameter('tcp_port', 5000)
        self.declare_parameter('tcp_host', '0.0.0.0')
        self.declare_parameter('max_speed', 255)

        self.tcp_port = self.get_parameter('tcp_port').get_parameter_value().integer_value
        self.tcp_host = self.get_parameter('tcp_host').get_parameter_value().string_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().integer_value

        self.observation_lock = threading.Lock()
        self.observation_cache = {
            'tracks.left_vel': 0.0,
            'tracks.right_vel': 0.0,
            'joint1.pos': 0.0,
            'joint2.pos': 0.0,
            'joint3.pos': 0.0,
            'joint4.pos': 0.0,
            'joint5.pos': 0.0,
            'wrist_cam': None,
            'front_cam': None,
        }

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/current_joint_state',
            self.joint_state_callback,
            10
        )

        self.wrist_cam_sub = self.create_subscription(
            CompressedImage,
            '/wrist_cam/image_raw/compressed',
            self.wrist_cam_callback,
            10
        )

        self.front_cam_sub = self.create_subscription(
            CompressedImage,
            '/front_cam/image_raw/compressed',
            self.front_cam_callback,
            10
        )

        self.motor_command_sub = self.create_subscription(
            MotorCommand,
            '/motor_command',
            self.motor_command_callback,
            10
        )

        self.motor_pub = self.create_publisher(MotorCommand, '/motor_command', 10)
        self.arm_delta_pub = self.create_publisher(ArmStateDelta, '/arm_state_delta', 10)
        self.servo_pub = self.create_publisher(ServoCommand, '/servo_command', 10)

        TCPRequestHandler.node = self

        self.tcp_server = socketserver.ThreadingTCPServer(
            (self.tcp_host, self.tcp_port),
            TCPRequestHandler
        )
        self.tcp_server.allow_reuse_address = True

        self.server_thread = threading.Thread(target=self.tcp_server.serve_forever, daemon=True)
        self.server_thread.start()

        self.get_logger().info(f'TCP Interface Node initialized on {self.tcp_host}:{self.tcp_port}')

    def joint_state_callback(self, msg: JointState):
        with self.observation_lock:
            if len(msg.position) >= 5:
                self.observation_cache['joint1.pos'] = float(msg.position[0])
                self.observation_cache['joint2.pos'] = float(msg.position[1])
                self.observation_cache['joint3.pos'] = float(msg.position[2])
                self.observation_cache['joint4.pos'] = float(msg.position[3])
                self.observation_cache['joint5.pos'] = float(msg.position[4])

    def wrist_cam_callback(self, msg: CompressedImage):
        try:
            with self.observation_lock:
                self.observation_cache['wrist_cam'] = bytes(msg.data)
        except Exception as e:
            self.get_logger().error(f'Error caching wrist camera image: {e}')

    def front_cam_callback(self, msg: CompressedImage):
        try:
            with self.observation_lock:
                self.observation_cache['front_cam'] = bytes(msg.data)
        except Exception as e:
            self.get_logger().error(f'Error caching front camera image: {e}')

    def motor_command_callback(self, msg: MotorCommand):
        with self.observation_lock:
            normalized_vel = self._motor_command_to_normalized(msg)

            if msg.motor_id == MotorCommand.MOTOR_LEFT:
                self.observation_cache['tracks.left_vel'] = normalized_vel
            elif msg.motor_id == MotorCommand.MOTOR_RIGHT:
                self.observation_cache['tracks.right_vel'] = normalized_vel

    def _motor_command_to_normalized(self, msg: MotorCommand) -> float:
        if msg.action == MotorCommand.ACTION_STOP:
            return 0.0

        normalized = msg.speed / float(self.max_speed)

        if msg.action == MotorCommand.ACTION_REVERSE:
            normalized = -normalized

        return np.clip(normalized, -1.0, 1.0)

    def get_observation(self) -> Dict[str, Any]:
        with self.observation_lock:
            obs = {}

            obs['tracks.left_vel'] = self.observation_cache['tracks.left_vel']
            obs['tracks.right_vel'] = self.observation_cache['tracks.right_vel']
            obs['joint1.pos'] = self.observation_cache['joint1.pos']
            obs['joint2.pos'] = self.observation_cache['joint2.pos']
            obs['joint3.pos'] = self.observation_cache['joint3.pos']
            obs['joint4.pos'] = self.observation_cache['joint4.pos']
            obs['joint5.pos'] = self.observation_cache['joint5.pos']

            if self.observation_cache['wrist_cam'] is not None:
                obs['wrist_cam'] = base64.b64encode(self.observation_cache['wrist_cam']).decode('utf-8')
            else:
                obs['wrist_cam'] = None

            if self.observation_cache['front_cam'] is not None:
                obs['front_cam'] = base64.b64encode(self.observation_cache['front_cam']).decode('utf-8')
            else:
                obs['front_cam'] = None

            return obs

    def send_action(self, action_data: Dict[str, float]):
        left_vel = np.clip(action_data.get('tracks.left_vel', 0.0), -1.0, 1.0)
        right_vel = np.clip(action_data.get('tracks.right_vel', 0.0), -1.0, 1.0)

        self.publish_motor_command(left_vel, right_vel)

        arm_delta_data = {
            'pos_x': np.clip(action_data.get('arm.pos.delta_x', 0.0), -1.0, 1.0),
            'pos_y': np.clip(action_data.get('arm.pos.delta_y', 0.0), -1.0, 1.0),
            'pos_z': np.clip(action_data.get('arm.pos.delta_z', 0.0), -1.0, 1.0),
            'rot_x': np.clip(action_data.get('arm.rot.delta_x', 0.0), -1.0, 1.0),
            'rot_y': np.clip(action_data.get('arm.rot.delta_y', 0.0), -1.0, 1.0),
            'rot_z': np.clip(action_data.get('arm.rot.delta_z', 0.0), -1.0, 1.0),
            'gripper': np.clip(action_data.get('arm.gripper.delta', 0.0), -1.0, 1.0),
        }

        self.publish_arm_delta(arm_delta_data)

    def publish_motor_command(self, left_vel: float, right_vel: float):
        left_snapped = self._snap_velocity(left_vel)
        right_snapped = self._snap_velocity(right_vel)

        self._send_motor_command(MotorCommand.MOTOR_LEFT, left_snapped)
        self._send_motor_command(MotorCommand.MOTOR_RIGHT, right_snapped)

    def _snap_velocity(self, vel: float) -> int:
        if vel > 0.33:
            return 1
        elif vel < -0.33:
            return -1
        else:
            return 0

    def _send_motor_command(self, motor_id: int, snapped_vel: int):
        command = MotorCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.motor_id = motor_id

        if snapped_vel == 0:
            command.action = MotorCommand.ACTION_STOP
            command.speed = 0
        elif snapped_vel > 0:
            command.action = MotorCommand.ACTION_FORWARD
            command.speed = self.max_speed
        else:
            command.action = MotorCommand.ACTION_REVERSE
            command.speed = self.max_speed

        self.motor_pub.publish(command)

    def publish_arm_delta(self, arm_data: Dict[str, float]):
        arm_delta = ArmStateDelta()
        arm_delta.translation = Vector3(
            x=arm_data['pos_x'],
            y=arm_data['pos_y'],
            z=arm_data['pos_z']
        )
        arm_delta.orientation = Vector3(
            x=arm_data['rot_x'],
            y=arm_data['rot_y'],
            z=arm_data['rot_z']
        )
        arm_delta.gripper = arm_data['gripper']

        self.arm_delta_pub.publish(arm_delta)

    def destroy_node(self):
        self.get_logger().info('Shutting down TCP server...')
        self.tcp_server.shutdown()
        self.tcp_server.server_close()
        super().destroy_node()


class TCPRequestHandler(socketserver.BaseRequestHandler):
    node: Optional[TCPInterfaceNode] = None

    def handle(self):
        while True:
            try:
                data = self.request.recv(1024*1024*10).decode('utf-8')

                if not data:
                    break

                request = json.loads(data)
                command = request.get('command')

                if command == 'get_observation':
                    response = self._handle_get_observation()
                elif command == 'send_action':
                    response = self._handle_send_action(request.get('data', {}))
                else:
                    response = {
                        'status': 'error',
                        'message': f'Unknown command: {command}'
                    }

                response_str = json.dumps(response)
                self.request.sendall(response_str.encode('utf-8'))

            except json.JSONDecodeError as e:
                error_response = {
                    'status': 'error',
                    'message': f'Invalid JSON: {str(e)}'
                }
                try:
                    self.request.sendall(json.dumps(error_response).encode('utf-8'))
                except:
                    break
            except Exception as e:
                self.node.get_logger().error(f'Error handling TCP request: {e}')
                error_response = {
                    'status': 'error',
                    'message': str(e)
                }
                try:
                    self.request.sendall(json.dumps(error_response).encode('utf-8'))
                except:
                    break

    def _handle_get_observation(self) -> Dict[str, Any]:
        try:
            obs = self.node.get_observation()
            return {
                'status': 'success',
                'data': obs
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to get observation: {str(e)}'
            }

    def _handle_send_action(self, action_data: Dict[str, float]) -> Dict[str, Any]:
        try:
            self.node.send_action(action_data)
            return {
                'status': 'success',
                'message': 'Action sent successfully'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to send action: {str(e)}'
            }
    
    # add handle_reset

def main(args=None):
    rclpy.init(args=args)
    node = TCPInterfaceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
