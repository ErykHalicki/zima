import numpy as np
import yaml

# need knowledge of arm structure (yaml file isntead of urdf)
# load arm structure from yaml on __init__(yaml_path)
# generate transformation matrices on init
# calculate pseudoinverse on init
# on solve() call take in xyzrpy in meters and radians
# use pre calculated pinv to get joint state output
# add biases to joint states
# return joint state in radians

class IKSolver:
    def __init__(self, arm_structure_yaml_path):
        with open(arm_structure_yaml_path, 'r') as f:
            structure_data = yaml.safe_load(f)
        arm_links = structure_data['links']
        for link in arm_links:
            print(link)
