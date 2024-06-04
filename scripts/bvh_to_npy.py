import glob
import pdb

from sklearn.pipeline import Pipeline
import os
from pymo.parsers import BVHParser
from pymo.preprocessing import *

target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3',
                 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist',
                 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist',
                 'b_neck0', 'b_head']


def get_joint_tree(path):
    p = BVHParser()
    X = p.parse(path)

    joint_name_to_idx = {}
    for i, joint in enumerate(X.traverse()):
        joint_name_to_idx[joint] = i

    # traverse tree
    joint_links = []
    stack = [X.root_name]
    while stack:
        joint = stack.pop()
        parent = X.skeleton[joint]['parent']
        # tab = len(stack)
        # print('%s- %s (%s)'%('| '*tab, joint, parent))
        if parent:
            joint_links.append((joint_name_to_idx[parent], joint_name_to_idx[joint]))
        for c in X.skeleton[joint]['children']:
            stack.append(c)

    print(joint_name_to_idx)
    print(joint_links)


def process_bvh(gesture_filename):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=30, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        # ('jtsel', JointSelector(target_joints, include_root=True)),
        ('param', MocapParameterizer('position')),        # expmap, position
        ('cnst', ConstantsRemover()),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    out_data = out_data[0]

    return out_data


def bvh_to_npy(bvh_path, sav_path):
    print(bvh_path)
    pos_data = process_bvh(bvh_path)
    # pos_data = np.pad(pos_data, ((0, 0), (3, 0)), 'constant', constant_values=(0, 0))
    print(pos_data.shape)
    npy_path = os.path.join(sav_path, bvh_path.split('\\')[-1].replace('.bvh', '.npy'))
    print(npy_path)
    np.save(npy_path, pos_data)


if __name__ == '__main__':
    # final model for test set
    # bvh_dir = "../my_inference/bvh/bvh_tst_for_twh_model/"
    # save_dir = "../my_inference/bvh2npy/UGTWH/"

    #  model for test set
    bvh_dir = "../my_inference/bvh/bvh_tst_for_ted_model/"
    save_dir = "../my_inference/bvh2npy/UGTED/"

    # real human model
    # bvh_dir = "../my_inference/bvh/bvh_UNA/"
    # save_dir = "../my_inference/bvh2npy/UGT"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # parse bvh
    files = sorted([f for f in glob.iglob(bvh_dir + '*.bvh')])
    print(files)
    for bvh_path in files:
        # print(bvh_path)
        bvh_to_npy(bvh_path, save_dir)
