from kinematic_solver import KinematicSolver
import numpy as np
import math
from matplotlib import pyplot as plt 

yaml_path = "/home/eryk/Documents/projects/zima/src/zima_ros2/src/zima_controls/arm_data/4dof_arm_structure.yaml"

num_arms = 1000
gen = np.random.default_rng()
ks = [100,500, 1000,2000,5000,10000,20000]
ks_errors = []

for k in ks:
    solver = KinematicSolver(yaml_path, k)
    random_samples = gen.random((num_arms,3))*math.pi - math.pi/2
    errors = []
    for true_joint_angles in random_samples:
        true_transformations = solver.forward(true_joint_angles)
        true_joint_positions = [solver.transformation_matrix_to_translation(T) for T in true_transformations]
        true_end_xyz = solver.transformation_matrix_to_translation(true_transformations[-1])
        true_end_rpy = solver.transformation_matrix_to_euler(true_transformations[-1])

        ik_joint_angles, ik_error = solver.solve(true_end_xyz, true_end_rpy)
        ik_joint_positions = [solver.transformation_matrix_to_translation(T) for T in solver.forward(ik_joint_angles)]
        errors.append(ik_error)
    ks_errors.append(errors)

fig, ax = plt.subplots()
ax.plot(ks, np.mean(ks_errors, axis=1))
ax.set_title(f'Mean ik error vs K')
ax.set_xlabel('K')
ax.set_ylabel('Mean Error')
plt.show()
