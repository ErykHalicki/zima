import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import math
import time

class KinematicSolver:
    def __init__(self, arm_structure_yaml_path, fk_sample_count = 20000):
        with open(arm_structure_yaml_path, 'r') as f:
            structure_data = yaml.safe_load(f)
        self.links = structure_data['links']
        self.revolute_links = [link for link in self.links if link['type']=='revolute']
        self.orientation_weight = 0.1
        self.translation_weight = 1.0
        self.kd_precision = 2
        self.kd_tree, self.joint_LUT = self.create_kd_tree(fk_sample_count)
        
    def forward(self,joints):
        if len(joints) != len(self.revolute_links):
            raise Exception("Cannot do FK with more joint angles than revolute links")

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

    def solve(self,xyz, rpy):
        '''
        xyz: numpy array of desired xyz
        rpy: numpy array of desired roll pitch yaw
        '''
        weighted_xyzrpy = np.concatenate((xyz*self.translation_weight, rpy*self.orientation_weight))
        _, best_point_index = self.kd_tree.query(weighted_xyzrpy)
        print(best_point_index)
        closest_xyzrpy = self.kd_tree.data[best_point_index]
        print(self.joint_LUT[tuple(closest_xyzrpy)])

    def create_kd_tree(self, k):
        '''
        k: number of points to randomly sample
        returns: kd_tree, LUT mapping weighted xyzrpy points to joint angles
        '''
        random_gen_start = time.time()
        gen = np.random.default_rng()
        random_samples = gen.random((k,len(self.revolute_links)))*math.pi 
        # random_samples is (k, d) matrix where each row is a set of randomly sampled joint states
        print(f"random gen took {time.time() - random_gen_start}")
    
        joint_LUT = {} # {(x,y,z,r,p,y): [joint0, joint1, ...]}
        kd_tree_data = np.empty((k,6)) #xyzrpy is 6dof 
        
        fk_start = time.time()
        for i, sample in enumerate(random_samples):
            T = self.forward(sample)[-1]
            weighted_eulers = self.transformation_matrix_to_euler(T) * self.orientation_weight
            weighted_translation = self.transformation_matrix_to_translation(T) * self.translation_weight
            weighted_xyzrpy = np.round(np.concatenate((weighted_translation, weighted_eulers)), self.kd_precision)
            joint_LUT[tuple(weighted_xyzrpy)] = sample
            kd_tree_data[i] = weighted_xyzrpy
        print(f"fk sampling took {time.time() - fk_start}")
        
        kd_start = time.time()
        kd_tree = KDTree(kd_tree_data)
        print(f"kd_tree generation took {time.time() - kd_start}")
        return kd_tree, joint_LUT

    def transformation_matrix_to_euler(self, matrix):
        R = matrix[:3, :3]
        r = Rotation.from_matrix(R)
        return r.as_euler('xyz', degrees=False)

    def transformation_matrix_to_translation(self, matrix):
        return matrix[:3,3]

if __name__ == '__main__':
    yaml_path = "/home/eryk/Documents/projects/zima/src/zima_ros2/src/zima_controls/arm_data/4dof_arm_structure.yaml"
    solver = KinematicSolver(yaml_path)
    solver.solve(np.array([0.1,0.1,0.1]), np.array([0,math.pi/2,0]))
    
