import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from zima_msgs.srv import SolveIK
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
import time
import argparse
import numpy as np

class IKClient(Node):
    def __init__(self, num_requests, max_x, max_y, max_z):
        super().__init__('ik_client')
        self.cli = self.create_client(SolveIK, 'solve_inverse_kinematics')
        
        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('service not available, waiting again...')
        
        self.num_requests = num_requests
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
    
    def generate_random_pose(self):
        # Generate a random pose within specified workspace bounds
        pose = Pose()
        pose.position = Point(
            x=np.random.uniform(0, self.max_x),
            y=np.random.uniform(-self.max_y, self.max_y),
            z=np.random.uniform(-self.max_z, self.max_z)
        )
        
        # Use identity quaternion (no rotation)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        return pose
    
    def send_request(self, pose):
        # Prepare initial guess (zero joint states)
        initial_guess = JointState()
        initial_guess.position = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Prepare request
        request = SolveIK.Request()
        request.target_pose = pose
        request.initial_guess = initial_guess
        
        # Send async request with timeout
        future = self.cli.call_async(request)
        
        # Wait for response with timeout
        rclpy.spin_until_future_complete(
            self, 
            future, 
            timeout_sec=5.0
        )
        
        # Check if future is done and return response
        if future.done():
            try:
                response = future.result()
                return response
            except Exception as e:
                self.get_logger().error(f'Service call failed: {str(e)}')
                return None
        else:
            self.get_logger().warn('Service call timed out')
            return None

def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='IK Service Performance Tester')
    parser.add_argument('-n', '--num_requests', type=int, default=10, 
                        help='Number of requests to send')
    parser.add_argument('--max_x', type=float, default=0.3, 
                        help='Maximum X coordinate')
    parser.add_argument('--max_y', type=float, default=0.3, 
                        help='Maximum Y coordinate')
    parser.add_argument('--max_z', type=float, default=0.3, 
                        help='Maximum Z coordinate')
    cli_args = parser.parse_args()

    rclpy.init(args=None)
    
    # Create client
    client = IKClient(
        cli_args.num_requests, 
        cli_args.max_x, 
        cli_args.max_y, 
        cli_args.max_z
    )
    
    # Record start time
    start_time = time.time()
    
    # Success and failure counters
    success_count = 0
    failure_count = 0
    timeout_count = 0
    
    # Send multiple requests
    for i in range(client.num_requests):
        pose = client.generate_random_pose()
        response = client.send_request(pose)
        
        # Track success/failure/timeout
        if response is None:
            timeout_count += 1
            print(f"Request {i+1}: Timed Out")
        elif response.success:
            success_count += 1
            print(f"Request {i+1}: Solved. Joint states: {response.solution.position}")
        else:
            failure_count += 1
            print(f"Request {i+1}: Failed")
    
    # Calculate and print performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n--- Performance Summary ---")
    print(f"Total Requests: {client.num_requests}")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests: {failure_count}")
    print(f"Timed Out Requests: {timeout_count}")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Average Time per Request: {total_time/client.num_requests:.4f} seconds")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
