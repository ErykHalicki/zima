import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider


def _draw_safety_boxes(ax, boxes):
    for box in boxes:
        x1, y1, z1 = box['x1'], box['y1'], box['z1']
        x2, y2, z2 = box['x2'], box['y2'], box['z2']

        vertices = [
            [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
            [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]
        ]

        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]

        poly = Poly3DCollection(faces, alpha=0.15, facecolor='red', edgecolor='darkred', linewidth=0.5)
        ax.add_collection3d(poly)


def visualize_arm(arms, safety_boxes=None, display_time=None):
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

    if safety_boxes:
        _draw_safety_boxes(ax, safety_boxes)

    max_range = 0.2
    ax.set_xlim([-max_range/2, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range/2, max_range])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Robot Arm Visualization')

    if display_time is not None:
        plt.show(block=False)
        plt.pause(display_time)
        plt.close(fig)
    else:
        plt.show()


def visualize_interactive(solver, initial_joints=None, joint_mode=False):
    if joint_mode:
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

            if hasattr(solver, 'safety_boxes'):
                _draw_safety_boxes(ax, solver.safety_boxes)

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

    else:
        initial_pose = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
        joints = [0.0] * len(solver.revolute_links)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.35)

        def update_arm_ik(pose):
            nonlocal joints
            ax.clear()
            joints, error, iterations = solver.solve(np.array(pose[:3]), np.array(pose[3:]), joints)
            if joints is not None:
                transforms = solver.forward(joints)
                points = np.array([t[:3, 3] for t in transforms])

                xs = points[:, 0]
                ys = points[:, 1]
                zs = points[:, 2]

                ax.plot(xs, ys, zs, 'o-', linewidth=2, markersize=8)
                ax.scatter(xs[0], ys[0], zs[0], c='green', s=100, label='Base')
                ax.scatter(xs[-1], ys[-1], zs[-1], c='red', s=100, label='End Effector')
                ax.scatter(pose[0], pose[1], pose[2], c='blue', s=100, marker='x', label='Target')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()
                ax.set_title(f'Target: ({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}) | Error: {error:.6f} | Iterations: {iterations}')
            else:
                ax.scatter(pose[0], pose[1], pose[2], c='blue', s=100, marker='x', label='Target')
                ax.set_title('IK Solution Not Found')

            if hasattr(solver, 'safety_boxes'):
                _draw_safety_boxes(ax, solver.safety_boxes)

            max_range = 0.3
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

            fig.canvas.draw_idle()

        sliders = []
        slider_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        slider_ranges = [(0.0, 0.30), (-0.3, 0.3), (-0.3, 0.3), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

        for i in range(6):
            ax_slider = plt.axes([0.1, 0.25 - i*0.04, 0.8, 0.02])
            slider = Slider(ax_slider, slider_labels[i], slider_ranges[i][0], slider_ranges[i][1], valinit=initial_pose[i])
            sliders.append(slider)

        def update(val):
            pose = [s.val for s in sliders]
            update_arm_ik(pose)

        for slider in sliders:
            slider.on_changed(update)

        update_arm_ik(initial_pose)
        plt.show()
