#include <stdio.h>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "kdl/tree.hpp"
#include "kdl/chain.hpp"
#include "kdl/frames.hpp"
#include "kdl/jntarray.hpp"
#include "kdl/chainiksolverpos_lma.hpp"
#include "kdl_parser/kdl_parser.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <random>
using std::placeholders::_1;
using std::vector;

class IKSolver : public rclcpp::Node
{
public:
    IKSolver() : Node("inverse_kinematics_node"){
        std::random_device rd; // Non-deterministic generator
        std::mt19937 gen(rd()); // Mersenne Twister generator
        std::uniform_real_distribution<> dis(-45.0, 45.0);
        
        target_pose_subscription = this->create_subscription<geometry_msgs::msg::Pose>
          ("/arm_target_pose", 10, std::bind(&IKSolver::target_pose_callback, this, _1));
        joint_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);

        kdl_parser::treeFromFile("robot.urdf",tree_);

        std::cout << "nb joints:        " << tree_.getNrOfJoints() << std::endl;
        std::cout << "nb segments:      " << tree_.getNrOfSegments() << std::endl;
        std::cout << "root segment:     " << tree_.getRootSegment()->first << std::endl;

        tree_.getChain("base_link", "palm", chain_);
        std::cout << "chain nb joints:  " << chain_.getNrOfJoints() << std::endl;
        
        unsigned int num_joints = chain_.getNrOfJoints();
        
        for(unsigned int i = 0; i < num_joints; i++) current_arm_angles.push_back(dis(gen)); //initialize starting position of all joints to non zero
        //non zero starting joints supposedly help

        KDL::JntArray q_min(num_joints); // Lower limits
        KDL::JntArray q_max(num_joints); // Upper limits

        // Set joint limits (replace with actual values)
        for (unsigned int i = 0; i < num_joints; i++) {
            q_min(i) = min_joint_angles[i] * M_PI / 180.0;
            q_max(i) = max_joint_angles[i] * M_PI / 180.0;
        }

        solver_ = new KDL::ChainIkSolverPos_LMA(chain_);
    }

private:
    void target_pose_callback(geometry_msgs::msg::Pose msg){
        //call solver code here
        RCLCPP_INFO(this->get_logger(), "received request");
        vector<float> angles = solveJointAngles(KDL::Vector(msg.position.x, msg.position.y, msg.position.z));
        
        sensor_msgs::msg::JointState joint_state_msg;
        joint_state_msg.header.stamp = this->get_clock()->now();

        for (size_t i = 0; i < angles.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "Joint %zu angle: %f", i, angles[i]);
            joint_state_msg.position.push_back(angles[i]);
        }
        joint_state_msg.name.push_back("base");
        joint_state_msg.name.push_back("shoulder");
        joint_state_msg.name.push_back("elbow");
        joint_state_msg.name.push_back("wrist");
        joint_state_msg.name.push_back("hand");
        //joint_state_msg.name.push_back("finger1");
        //joint_state_msg.name.push_back("finger2");

        // Publish joint state message
        joint_publisher_->publish(joint_state_msg);
    }
    
    vector<float> solveJointAngles(KDL::Vector pos){ //returns desired angles in degrees
        // Prepare IK solver input variables
        vector<float> result;

        KDL::JntArray q_init(chain_.getNrOfJoints());

        for(unsigned int i = 0; i<chain_.getNrOfJoints(); i++) q_init(i) = current_arm_angles[i] * M_PI/180;

        const KDL::Frame p_in(pos);
        KDL::JntArray q_out(chain_.getNrOfJoints());

        // Run IK solver
        solver_->CartToJnt(q_init, p_in, q_out);

        for(unsigned int i = 0; i<chain_.getNrOfJoints(); i++) {
            result.push_back(q_out(i) * 180/M_PI);
            current_arm_angles[i] = result[i];
        }
        return result;
    }

    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr target_pose_subscription;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_publisher_;
    KDL::Tree tree_;
    KDL::Chain chain_;
    KDL::ChainIkSolverPos_LMA* solver_;
    //KDL::ChainIkSolverPos_NR* solver_;
    //KDL::ChainFkSolverPos_recursive* fk_solver_;
    //KDL::ChainIkSolverVel_pinv* ik_solver_;
    vector<float> current_arm_angles;
    vector<double> min_joint_angles = {-90, -90,-90,-90,-90,-90,-90,-90};  // Replace with actual limits
    vector<double> max_joint_angles = {90,90,90,90,90,90,90,90};  
   
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IKSolver>());
  rclcpp::shutdown();
  return 0;
}
