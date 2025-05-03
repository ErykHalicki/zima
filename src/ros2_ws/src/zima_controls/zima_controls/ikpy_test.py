from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

def main():
    my_chain = Chain.from_urdf_file("robot_arm_ikpy.urdf")
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    current_pose = [0,0,0,0,0,0,0] #first and last joint are palceholders used by ikpy, only middle 5 correspond to zima arm
    for i in range(0,20):
        ax.clear()  # Clear the axis between each plot
        desired_pose = my_chain.forward_kinematics(current_pose)
        my_chain.plot(my_chain.inverse_kinematics_frame(desired_pose, current_pose), ax)
        ax.set_title(f'Pose {i+1}: {desired_pose}')
        ax.set_xlim(-0.5, 0.5)  # Set consistent axis limits
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.5)
        current_pose[3] += 0.08
        current_pose[4] += 0.08
        matplotlib.pyplot.draw()
        matplotlib.pyplot.pause(0.2)  # Pause to update the plot
    
    matplotlib.pyplot.show()  # Keep the window open after all iterations

if __name__ == '__main__':
    main()
