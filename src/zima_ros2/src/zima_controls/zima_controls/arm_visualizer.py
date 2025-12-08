import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider


def visualize_arm(arms):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, len(arms)))

    for i, arm_joints in enumerate(arms):
        points = np.array(arm_joints)
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        color = colors[i]
        ax.plot(xs, ys, zs, 'o-', linewidth=2, markersize=8, color=color, label=f'Arm {i+1}')
        ax.scatter(xs[0], ys[0], zs[0], c='green', s=100)
        ax.scatter(xs[-1], ys[-1], zs[-1], c='red', s=100, edgecolors=color, linewidths=2)

    max_range = 0.2
    ax.set_xlim([-max_range/2, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range/2, max_range])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Robot Arm Visualization')

    plt.show()


def visualize_interactive(solver, initial_joints=None):
    if initial_joints is None:
        initial_joints = [0.0] * len(solver.revolute_links)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    def update_arm(joints):
        ax.clear()
        transforms = solver.forward(joints)
        points = np.array([t[:3, 3] for t in transforms])

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        ax.plot(xs, ys, zs, 'o-', linewidth=2, markersize=8)
        ax.scatter(xs[0], ys[0], zs[0], c='green', s=100, label='Base')
        ax.scatter(xs[-1], ys[-1], zs[-1], c='red', s=100, label='End Effector')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'End Effector: ({xs[-1]:.3f}, {ys[-1]:.3f}, {zs[-1]:.3f})')

        max_range = 0.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        fig.canvas.draw_idle()

    sliders = []
    for i in range(len(solver.revolute_links)):
        ax_slider = plt.axes([0.1, 0.15 - i*0.03, 0.8, 0.02])
        slider = Slider(ax_slider, f'Joint {i+1}', -np.pi, np.pi, valinit=initial_joints[i])
        sliders.append(slider)

    def update(val):
        joints = [s.val for s in sliders]
        update_arm(joints)

    for slider in sliders:
        slider.on_changed(update)

    update_arm(initial_joints)
    plt.show()
