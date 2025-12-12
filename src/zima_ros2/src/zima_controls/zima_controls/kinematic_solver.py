import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import math
import time

class KinematicSolver:
    def __init__(self, arm_structure_yaml_path, arm_safety_yaml_path=None, fk_sample_count = 10000, dimension_mask=[1]*6):
        '''
        dimension_mask (xyzrpy): specifies which dimensions to optimize against during inverse kin solving
        '''
        with open(arm_structure_yaml_path, 'r') as f:
            structure_data = yaml.safe_load(f)
        with open(arm_safety_yaml_path, 'r') as f:
            safety_data = yaml.safe_load(f)
        self.safety_boxes = None
        if arm_safety_yaml_path is not None:
            self.safety_boxes = safety_data['boxes']
        self.dimension_mask = dimension_mask
        self.links = structure_data['links']
        self.revolute_links = [link for link in self.links if link['type']=='revolute']
        self.orientation_weight = 0.05
        self.translation_weight = 1.0
        self.joint_weight = 0.1
        self.kd_precision = 4
        self.kd_tree, self.joint_LUT = self.create_kd_tree(fk_sample_count)
        self.max_reach = sum([np.linalg.norm(link['translation']) for link in self.links])
        
    def clamp_joints(self, joints, revolute_only=True, with_bias=False):
        if revolute_only and len(joints) != len(self.revolute_links):
            raise Exception(f"Cannot clamp {len(joints)} revolute joints, must pass in {len(self.revolute_links)}")
        elif not revolute_only and len(joints) != len(self.links):
            raise Exception(f"Cannot clamp {len(joints)} joints, must pass in {len(self.links)}")
        
        result = []
        links=None

        if revolute_only:
            links=self.revolute_links
        else:
            links=self.links

        for i, link in enumerate(links):
            bias=0.0
            if with_bias:
                bias = link['bias']
            result.append(min(link['max']+bias, max(joints[i], link['min']+bias)))
        return result

    def forward(self,joints):
        '''
        joints: list of revolute joint states
        '''
        if len(joints) != len(self.revolute_links):
            print(f"Joints: {joints}")
            raise Exception(f"Cannot do FK with {len(joints)} joint angles, there are {len(self.revolute_links)} revolute links")
        joints = self.clamp_joints(joints)
        T = [np.eye(4)]
        i=0
        for link in self.links:
            axis = np.array(link['axis'])/np.linalg.norm(link['axis'])
            if link['type'] == 'revolute':
                R = Rotation.from_rotvec((joints[i] + link['bias']) * axis)
                i+=1
            elif link['type'] == 'fixed':
                R = Rotation.from_rotvec(link['bias'] * axis)
            T_link = np.eye(4)
            T_link[:3,:3] = R.as_matrix()
            T_link[:3,3] = link['translation']
            T.append(T[-1]@T_link)
        return T

    def calculate_joint_state_error(self, joint_state, target_xyz, target_rpy, current_joint_state=None):
        T = self.forward(joint_state)[-1]
        weighted_orientation_errors = (target_rpy - self.transformation_matrix_to_euler(T)) * self.orientation_weight
        weighted_translation_errors = (target_xyz - self.transformation_matrix_to_translation(T)) * self.translation_weight
        total_weighted_errors = np.concatenate((weighted_translation_errors, weighted_orientation_errors))
        total_weighted_errors*=self.dimension_mask
        if current_joint_state is not None:
            joint_errors = (np.array(current_joint_state) - joint_state) 
            weighted_joint_errors = joint_errors  * self.joint_weight #keep sign but punish large changes in just one axis
            total_weighted_errors = np.concatenate((total_weighted_errors, weighted_joint_errors))
        return total_weighted_errors

    def estimate_jacobian(self, joint_state, target_xyz, target_rpy, current_joint_state=None, h=np.radians(2)): # default of 2 degrees
        error_dim = 6 
        if current_joint_state is not None:
            error_dim += len(self.revolute_links)
        jacobian = np.empty((error_dim,len(joint_state)))
        for i in range(len(joint_state)):
            joint_state_plus = joint_state.copy()
            joint_state_minus = joint_state.copy()
            joint_state_plus[i] += h
            joint_state_minus[i] -= h
            plus_h_error = self.calculate_joint_state_error(joint_state_plus, target_xyz, target_rpy, current_joint_state)
            minus_h_error = self.calculate_joint_state_error(joint_state_minus, target_xyz, target_rpy, current_joint_state)
            jacobian[:,i] = (plus_h_error - minus_h_error) / (2*h) # ith column corresponds to the ith joints partial derivatives of error
        return jacobian

    def solve(self, xyz, rpy=[0,0,0], current_joint_state = None, eps=0.01, max_iters=20, step_size=0.1):
        '''
        xyz: numpy array of desired xyz
        rpy: numpy array of desired roll pitch yaw
        current_joint_state: if provided will also optimze for minimum joint difference
        '''
        if current_joint_state is not None and len(current_joint_state) != len(self.revolute_links):
            raise Exception(f"Cannot solve for more than {len(self.revolute_links)} joint states!")

        estimate = current_joint_state
        if current_joint_state is None:
            weighted_xyzrpy = np.concatenate((xyz*self.translation_weight, rpy*self.orientation_weight))
            warm_start_error, best_point_index = self.kd_tree.query(weighted_xyzrpy)
            closest_xyzrpy = self.kd_tree.data[best_point_index]
            estimate = self.joint_LUT[tuple(closest_xyzrpy)]
        
        i=0
        while i < max_iters:
            current_error = self.calculate_joint_state_error(estimate, xyz, rpy, current_joint_state)
            if np.linalg.norm(current_error) <= eps:
                break # solution is good enough
            jacobian = self.estimate_jacobian(estimate, xyz,rpy, current_joint_state)
            estimate += step_size * np.linalg.pinv(jacobian) @ -current_error # (joint_dim, error_dim) @ (error_dim, 1)
            estimate = self.clamp_joints(estimate)
            i+=1
        else:
            current_error = self.calculate_joint_state_error(estimate,xyz,rpy)
        return estimate, np.linalg.norm(current_error), i

    def is_joint_state_safe(self, joint_state):
        if self.safety_boxes is None:
            raise Exception("Cannot evaluate joint safety with no provided safety box file")
        T = self.forward(joint_state)
        joint_translations = [self.transformation_matrix_to_translation(t) for t in T]
        for box in self.safety_boxes:
            for j, joint_translation in enumerate(joint_translations):
                in_box = True
                for i, dim in enumerate(['x', 'y', 'z']):
                    below_max = joint_translation[i] < max(box[f'{dim}1'], box[f'{dim}2'])
                    above_min = joint_translation[i] > min(box[f'{dim}1'], box[f'{dim}2'])
                    if not (above_min and below_max):
                        in_box = False
                if in_box:
                    return False, self.links[j-1]['name']
        return True, None

    def create_kd_tree(self, k):
        '''
        k: number of points to randomly sample
        returns: kd_tree, LUT mapping weighted xyzrpy points to joint angles
        '''
        random_gen_start = time.time()
        gen = np.random.default_rng()
        random_samples = gen.random((k,len(self.revolute_links)))*math.pi - math.pi/2
        # random_samples is (k, d) matrix where each row is a set of randomly sampled joint states
        #print(f"random gen took {time.time() - random_gen_start}")
    
        joint_LUT = {} # {(x,y,z,r,p,y): [joint0, joint1, ...]}
        kd_tree_data = np.empty((k,6)) #xyzrpy is 6dof

        fk_start = time.time()
        for i, sample in enumerate(random_samples):
            T = self.forward(sample)[-1]
            weighted_eulers = self.transformation_matrix_to_euler(T) * self.orientation_weight
            weighted_translation = self.transformation_matrix_to_translation(T) * self.translation_weight
            weighted_xyzrpy = np.round(np.concatenate((weighted_translation, weighted_eulers)), self.kd_precision)
            weighted_xyzrpy*=self.dimension_mask
            joint_LUT[tuple(weighted_xyzrpy)] = sample
            kd_tree_data[i] = weighted_xyzrpy
        print(len(joint_LUT))
        #print(f"fk sampling took {time.time() - fk_start}")
        kd_tree_data*= self.dimension_mask
        
        kd_start = time.time()
        kd_tree = KDTree(kd_tree_data)
        #print(f"kd_tree generation took {time.time() - kd_start}")
        return kd_tree, joint_LUT

    def transformation_matrix_to_euler(self, matrix):
        R = matrix[:3, :3]
        r = Rotation.from_matrix(R)
        return r.as_euler('xyz', degrees=False)

    def transformation_matrix_to_translation(self, matrix):
        return np.array(matrix[:3,3])

if __name__ == '__main__':
    from arm_visualizer import visualize_arm, visualize_interactive
    yaml_path = "/home/eryk/Documents/projects/zima/src/zima_ros2/src/zima_controls/arm_data/4dof_arm_structure.yaml"
    safety_path = "/home/eryk/Documents/projects/zima/src/zima_ros2/src/zima_controls/arm_data/4dof_arm_safety.yaml"

    solver = KinematicSolver(yaml_path, arm_safety_yaml_path=safety_path, dimension_mask=[1,1,1,1,0,0])
    
    '''
    num_arms = 20
    gen = np.random.default_rng()
    random_samples = gen.random((num_arms,4))*math.pi
    errors = []
    for true_joint_angles in random_samples:
        true_transformations = solver.forward(true_joint_angles)
        true_joint_positions = [solver.transformation_matrix_to_translation(T) for T in true_transformations]
        true_end_xyz = solver.transformation_matrix_to_translation(true_transformations[-1])
        true_end_rpy = solver.transformation_matrix_to_euler(true_transformations[-1])

        ik_joint_angles, ik_error = solver.solve(true_end_xyz, true_end_rpy)
        ik_joint_positions = [solver.transformation_matrix_to_translation(T) for T in solver.forward(ik_joint_angles)]
        visualize_arm([true_joint_positions, ik_joint_positions])
    '''

    visualize_interactive(solver, joint_mode=False)
    
