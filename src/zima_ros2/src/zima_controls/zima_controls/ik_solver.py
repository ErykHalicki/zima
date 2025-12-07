import numpy as np
import yaml
from scipy.spatial.transform import Rotation

class IKSolver:
    def __init__(self, arm_structure_yaml_path):
        with open(arm_structure_yaml_path, 'r') as f:
            structure_data = yaml.safe_load(f)
        self.links = structure_data['links']
        self.revolute_links = [link for link in self.links if link['type']=='revolute']
        
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

    def solve(self,xyz,rpy):
        pass
    

    def matrix_to_euler(matrix):
        R = matrix[0:3, 0:3]
        r = Rotation.from_matrix(R)
        return r.as_euler('xyz', degrees=False)

# on solve() call take in xyzrpy in meters and radians
# use pre calculated pinv to get joint state output
# add biases to joint states
# return joint state in radians

if __name__ == '__main__':
    from arm_visualizer import visualize_interactive

    yaml_path = "/Users/erykhalicki/Documents/projects/current/zima/src/zima_ros2/src/zima_controls/arm_data/4dof_arm_structure.yaml"
    solver = IKSolver(yaml_path)
    visualize_interactive(solver)
    
