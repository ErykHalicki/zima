import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from ikpy.chain import Chain
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from zima_msgs.srv import SolveIK
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

class IKService(Node):
    def __init__(self):
        super().__init__('ik_service')
        
        # Declare ROS2 parameters
        debug_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description='Enable debug visualization of IK solution'
        )
        self.declare_parameter('debug_mode', False, debug_descriptor)
        
        threshold_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Position error threshold for IK solution in meters'
        )
        self.declare_parameter('position_error_threshold', 0.03, threshold_descriptor)
        
        # Load the URDF chain
        self.chain = Chain.from_urdf_file("robot_arm_ikpy.urdf")
        
        # Create the service
        self.srv = self.create_service(SolveIK, 'solve_inverse_kinematics', self.solve_ik_callback)
    
    def visualize_arm_state(self, chain, joint_states):
        # Prepare visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepend and append placeholder joints for IKPy
        full_solution = [0] + list(joint_states) + [0]
        
        # Plot the chain
        chain.plot(full_solution, ax)
        
        # Set plot properties
        ax.set_title('IK Solution Arm State')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.5)
        
        # Show the plot
        plt.show()
    
    def quaternion_to_rotation_matrix(self, q):
        # Convert quaternion to rotation matrix
        x, y, z, w = q
        
        x2, y2, z2 = x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, yz, zx = x*y, y*z, z*x
        
        rotation_matrix = np.array([
            [1 - 2*(y2 + z2), 2*(xy - wz), 2*(zx + wy)],
            [2*(xy + wz), 1 - 2*(x2 + z2), 2*(yz - wx)],
            [2*(zx - wy), 2*(yz + wx), 1 - 2*(x2 + y2)]
        ])
        
        return rotation_matrix
    
    def pose_to_matrix(self, pose):
        # Convert ROS Pose to 4x4 transformation matrix
        position = np.array([
            pose.position.x, 
            pose.position.y, 
            pose.position.z
        ])
        
        # Convert quaternion to rotation matrix
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        
        # Combine into 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position
        
        return transform_matrix
    
    def matrix_to_pose(self, matrix):
        # Convert 4x4 transformation matrix to ROS Pose
        pose = Pose()
        
        # Extract position
        pose.position.x = matrix[0, 3]
        pose.position.y = matrix[1, 3]
        pose.position.z = matrix[2, 3]
        
        # Extract rotation (convert to quaternion)
        rot_matrix = matrix[:3, :3]
        
        # Quaternion calculation
        trace = np.trace(rot_matrix)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
            y = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
            z = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
        else:
            if rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                S = np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
                w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
                x = 0.25 * S
                y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
                z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
            elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                S = np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
                w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
                x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
                y = 0.25 * S
                z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
            else:
                S = np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
                w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
                x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
                y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
                z = 0.25 * S
        
        pose.orientation.x = x
        pose.orientation.y = y
        pose.orientation.z = z
        pose.orientation.w = w
        
        return pose
    
    def calculate_position_error(self, target_pose, achieved_pose):
        # Calculate position error (Euclidean distance)
        pos_error = np.linalg.norm([
            target_pose.position.x - achieved_pose.position.x,
            target_pose.position.y - achieved_pose.position.y,
            target_pose.position.z - achieved_pose.position.z
        ])
        
        return pos_error
    
    def solve_ik_callback(self, request, response):
        try:
            # Get current parameter values
            debug_mode = self.get_parameter('debug_mode').value
            pos_error_threshold = self.get_parameter('position_error_threshold').value
            
            # Convert pose to 4x4 transformation matrix
            target_matrix = self.pose_to_matrix(request.target_pose)
            
            # Use initial guess if provided, otherwise use zero position
            # Convert initial guess from degrees to radians
            '''
            initial_guess = [np.clip(np.radians(pos), np.radians(-89.9), np.radians(89.9)) for pos in request.initial_guess.position] if request.initial_guess.position else [0, 0, 0, 0, 0]
            
            for pos in request.initial_guess.position: #if the arm is not initialized
                if pos == -91.0:
                    initial_guess = [0,0,0,0,0]
                    break
            '''

            initial_guess = [0,0,0,0,0] #for some reason adding a non zero initial guess prevented solution with cooridnates in the neagtives (ex. x = -0.1)
            
            # Prepend and append placeholder joints for IKPy
            full_initial_guess = [0] + list(initial_guess) + [0]
            
            # Solve inverse kinematics
            ik_solution = self.chain.inverse_kinematics_frame(
                target=target_matrix, 
                initial_position=full_initial_guess,
                orientation_mode="all"
            )
            
            # Remove first and last placeholder joints
            joint_states = ik_solution[1:-1]
            
            # Do forward kinematics to verify
            full_fk_solution = [0] + list(joint_states) + [0]
            achieved_matrix = self.chain.forward_kinematics(full_fk_solution)
            
            # Convert achieved matrix to pose
            achieved_pose = self.matrix_to_pose(achieved_matrix)
            
            # Calculate position error
            pos_error = self.calculate_position_error(request.target_pose, achieved_pose)
            
            # Decide on success
            if pos_error <= pos_error_threshold:
                # Successful IK solution - convert to degrees and round
                joint_states_deg = [round(np.degrees(js), 1) for js in joint_states]
                
                # Successful IK solution
                response.solution.position = joint_states_deg
                response.success = True
                self.get_logger().info(f'IK solved: {joint_states_deg}. Pos error: {pos_error}')
                
                # Optional debug visualization
                if debug_mode:
                    self.visualize_arm_state(self.chain, joint_states)
            else:
                # Failed to find a good solution
                response.solution.position = initial_guess
                response.success = False
                self.get_logger().warning(f'IK solution failed. Pos error: {pos_error}')
            
            return response
        
        except Exception as e:
            self.get_logger().error(f'IK solving failed: {str(e)}')
            response.solution.position = request.initial_guess.position
            response.success = False
            return response

def main(args=None):
    rclpy.init(args=args)
    ik_service = IKService()
    rclpy.spin(ik_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
