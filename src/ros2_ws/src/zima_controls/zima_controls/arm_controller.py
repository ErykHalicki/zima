import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

from zima_msgs.srv import SolveIK


class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')
        
        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Current servo positions subscriber
        self.current_positions_sub = self.create_subscription(
            JointState, 
            '/current_servo_positions', 
            self.current_positions_callback, 
            qos_profile
        )
        
        # Goal arm pose subscriber
        self.goal_pose_sub = self.create_subscription(
            Pose, 
            '/goal_arm_pose', 
            self.goal_pose_callback, 
            qos_profile
        )
        
        # IK solver service client
        self.ik_client = self.create_client(SolveIK, 'solve_inverse_kinematics')
        
        # Goal joint state publisher
        self.goal_joint_state_pub = self.create_publisher(
            JointState, 
            '/goal_joint_state', 
            qos_profile
        )
        
        # Store current servo positions
        self.current_positions = None
        self.client_futures = []
        
        self.get_logger().info('Arm Controller node initialized')
    
    def current_positions_callback(self, msg):
        """Update current servo positions"""
        self.current_positions = msg
    
    def goal_pose_callback(self, goal_pose):
        """Solve IK for the goal pose and publish goal joint states"""
        # Wait for IK service to be available
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('IK service not available')
            return
        
        # Prepare IK service request
        request = SolveIK.Request()
        request.target_pose = goal_pose
        
        # Use current positions as initial guess if available
        if self.current_positions:
            request.initial_guess.position = self.current_positions.position
        
        # Call IK service
        #self.get_logger().info(f'{request}')
        self.client_futures.append(self.ik_client.call_async(request))

    def send_goal_joint_state(self, response):
        try:
            if response.success:
                # Create goal joint state message
                goal_joint_state = JointState()
                goal_joint_state.position = [float(pos) for pos in response.solution.position]
                goal_joint_state.header.stamp = self.get_clock().now().to_msg()
                
                # Publish goal joint states
                self.goal_joint_state_pub.publish(goal_joint_state)
                self.get_logger().info(f'Published goal joint states: {goal_joint_state.position}')
            else:
                self.get_logger().warning('IK solver failed to find a solution')
        
        except Exception as e:
            self.get_logger().error(f'Error processing IK solution: {str(e)}')

    def spin(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            incomplete_futures = []
            for f in self.client_futures:
                if f.done():
                    res = f.result()
                    self.send_goal_joint_state(res)
                else:
                    incomplete_futures.append(f)
            self.client_futures = incomplete_futures

def main(args=None):
    rclpy.init(args=args)
    arm_controller = ArmController()
    arm_controller.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
