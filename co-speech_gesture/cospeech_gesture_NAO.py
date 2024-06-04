#!/usr/bin/env python
import math
import numpy as np
import pandas as pd
import string
import argparse
import json
from naoqi import ALProxy
import time
import almath

# ############################################################
# FOR NAO ROBOT
NAO_TO_CMU_JOINT_MAPPING = {
    'HeadYaw': 'b_head.Zrotation',
    'HeadPitch': 'b_head.Yrotation',

    'LShoulderPitch': 'b_l_shoulder.Xrotation',  # Y
    'LShoulderRoll': 'b_l_shoulder.Yrotation',

    'LElbowYaw': 'b_l_arm.Zrotation',
    'LElbowRoll': 'b_l_forearm.Xrotation',

    'LWristYaw': 'b_l_wrist.Yrotation',

    'RShoulderPitch': 'b_r_shoulder.Xrotation',
    'RShoulderRoll': 'b_r_shoulder.Yrotation',

    'RElbowYaw': 'b_r_arm.Zrotation',
    'RElbowRoll': 'b_r_forearm.Xrotation',

    'RWristYaw': 'b_r_wrist.Yrotation'
}

CMU_TO_NAO_JOINT_MAPPING = {v: k for k, v in NAO_TO_CMU_JOINT_MAPPING.items()}

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

_NEXT_AXIS = [1, 2, 0, 1]
_EPS = np.finfo(np.float64).eps * 4.0


def euler_from_matrix(matrix, axes='sxzx'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (np.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not np.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


# Node
import itertools


class Node:
    def __init__(self, root=False):
        self.name = None
        self.channels = []
        self.offset = (0, 0, 0)
        self.children = []
        self.is_root = root

    def isEndSite(self):
        return len(self.children) == 0

    def to_json(self):
        return {
            "name": self.name,
            "channels": self.channels,
            "offset": self.offset,
            "is_root": self.is_root,
            "children": [child.to_json() for child in self.children]
        }

    def __repr__(self):
        return json.dumps(self.to_json(), indent=2)

    def get_node_info_string(self):
        return '{} [{}]'.format(self.name, ' '.join([channel for channel in self.channels]))

    def get_unique_node_info(self):
        unique_node_infos = [self.get_node_info_string()]
        unique_node_infos += itertools.chain.from_iterable([child.get_unique_node_info() for child in self.children])
        unique_node_infos = list(set(unique_node_infos))
        unique_node_infos.sort()
        return unique_node_infos


# BVHLoader
class BVHLoader:
    def __init__(self, filename):
        self.filename = filename  # BVH filename
        self.tokenlist = []  # A list of unprocessed tokens (strings)
        self.linenr = 0  # The current line number

        # Root node
        self.root = None
        self.nodestack = []

        # Total number of channels
        self.numchannels = 0

        # Motion
        self.all_motions = []
        self.dt = 1
        self.num_motions = 1

        self.counter = 0
        self.this_motion = None

        self.scaling_factor = 0.1

        # Read file
        self.fhandle = open(self.filename, 'r')
        self.readHierarchy()
        self.readMotion()

    # Tokenization function
    def readLine(self):
        """Return the next line."""
        s = self.fhandle.readline()
        self.linenr += 1
        if s == "":
            raise None  # End of file
        return s

    def token(self):
        """Return the next token."""
        # Are there still some tokens left? then just return the next one
        if self.tokenlist != []:
            tok = self.tokenlist[0]
            self.tokenlist = self.tokenlist[1:]
            return tok

        # Read a new line
        s = self.readLine()
        self.tokenlist = s.strip().split()
        return self.token()

    def intToken(self):
        """Return the next token which must be an int.
        """
        tok = self.token()
        try:
            return int(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Integer expected, got '%s' instead" % (self.linenr, tok))

    def floatToken(self):
        """Return the next token which must be a float.
        """
        tok = self.token()
        try:
            return float(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Float expected, got '%s' instead" % (self.linenr, tok))

    ###
    # Read Hierarchy
    ###
    def readHierarchy(self):
        """
        Read the skeleton hierarchy to self.nodestack and store root in self.root
        """
        tok = self.token()
        if tok != "HIERARCHY":
            raise SyntaxError("Syntax error in line %d: 'HIERARCHY' expected, got '%s' instead" % (self.linenr, tok))

        tok = self.token()
        if tok != "ROOT":
            raise SyntaxError("Syntax error in line %d: 'ROOT' expected, got '%s' instead" % (self.linenr, tok))

        self.root = Node(root=True)
        self.nodestack.append(self.root)
        self.readNode()

        # Handler On Hierarchy
        print(self.root.children[0].children[0])
        # self.scaling_factor = 0
        self.scaling_factor = 0.1 / self.root.children[0].children[0].offset[1]
        # if self.root.children[0].children[0].offset[0]==0.0:
        #     self.scaling_factor =
        # else:
        #     self.scaling_factor = 0.1 / self.root.children[0].children[0].offset[0]

    # readNode
    def readNode(self):
        """Recursive function to recursively read the data for a node.
        """
        # Read the node name (or the word 'Site' if it was a 'End Site' node)
        name = self.token()
        self.nodestack[-1].name = name

        tok = self.token()
        if tok != "{":
            raise SyntaxError("Syntax error in line %d: '{' expected, got '%s' instead" % (self.linenr, tok))

        while True:
            tok = self.token()
            if tok == "OFFSET":
                x, y, z = self.floatToken(), self.floatToken(), self.floatToken()
                self.nodestack[-1].offset = (x, y, z)
            elif tok == "CHANNELS":
                n = self.intToken()
                channels = []
                for i in range(n):
                    tok = self.token()
                    if tok not in ["Xposition", "Yposition", "Zposition",
                                   "Xrotation", "Yrotation", "Zrotation"]:
                        raise SyntaxError("Syntax error in line %d: Invalid channel name: '%s'" % (self.linenr, tok))
                    channels.append(tok)
                self.numchannels += len(channels)
                self.nodestack[-1].channels = channels
            elif tok == "JOINT":
                node = Node()
                self.nodestack[-1].children.append(node)
                self.nodestack.append(node)
                self.readNode()
            elif tok == "End":
                node = Node()
                self.nodestack[-1].children.append(node)
                self.nodestack.append(node)
                self.readNode()
            elif tok == "}":
                if self.nodestack[-1].isEndSite():
                    self.nodestack[-1].name = "End Site"
                self.nodestack.pop()
                break
            else:
                raise SyntaxError("Syntax error in line %d: Unknown keyword '%s'" % (self.linenr, tok))

                ###

    # Read Motion
    ###
    def readMotion(self):
        """Read the motion samples and stores to self.all_motions
        """
        # No more tokens (i.e. end of file)? Then just return
        tok = self.token()
        if not tok:
            return

        if tok != "MOTION":
            raise SyntaxError("Syntax error in line %d: 'MOTION' expected, got '%s' instead" % (self.linenr, tok))

        # Read the number of frames
        tok = self.token()
        if tok != "Frames:":
            raise SyntaxError("Syntax error in line %d: 'Frames:' expected, got '%s' instead" % (self.linenr, tok))

        frames = self.intToken()

        # Read the frame time
        tok = self.token()
        if tok != "Frame":
            raise SyntaxError("Syntax error in line %d: 'Frame Time:' expected, got '%s' instead" % (self.linenr, tok))
        tok = self.token()
        self.num_motions = frames

        if tok != "Time:":
            raise SyntaxError(
                "Syntax error in line %d: 'Frame Time:' expected, got 'Frame %s' instead" % (self.linenr, tok))

        dt = self.floatToken()
        # Handler OnMotion
        self.dt = dt
        print (self.dt)

        # Read the channel values
        for i in range(frames):
            s = self.readLine()
            # print s
            a = s.split()
            if len(a) != self.numchannels:
                raise SyntaxError("Syntax error in line %d: %d float values expected, got %d instead" % (
                    self.linenr, self.numchannels, len(a)))
            values = map(lambda x: float(x), a)

            # Handler OnFrame
            self.all_motions.append(list(values))

    # extractRootJoint
    def extractRootJoint(self, root, gesture_dict):
        if root.isEndSite():
            return gesture_dict

        # Calculate transformation for mapped joints
        num_channels = len(root.channels)
        flag_trans = 0
        flag_rot = 0
        rx, ry, rz = 0, 0, 0
        rot_mat = np.array([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])

        for channel in root.channels:
            keyval = self.this_motion[self.counter]
            if (channel == "Xrotation"):
                flag_rot = True
                xrot = keyval
                theta = math.radians(xrot)
                c = math.cos(theta)
                s = math.sin(theta)
                rot_mat_x = np.array([[1., 0., 0., 0.],
                                      [0., c, -s, 0.],
                                      [0., s, c, 0.],
                                      [0., 0., 0., 1.]])
                rot_mat = np.matmul(rot_mat, rot_mat_x)
            elif (channel == "Yrotation"):
                flag_rot = True
                yrot = keyval
                theta = math.radians(yrot)
                c = math.cos(theta)
                s = math.sin(theta)
                rot_mat_y = np.array([[c, 0., s, 0.],
                                      [0., 1., 0., 0.],
                                      [-s, 0., c, 0.],
                                      [0., 0., 0., 1.]])
                rot_mat = np.matmul(rot_mat, rot_mat_y)

            elif (channel == "Zrotation"):
                flaisRootg_rot = True
                zrot = keyval
                theta = math.radians(zrot)
                c = math.cos(theta)
                s = math.sin(theta)
                rot_mat_z = np.array([[c, -s, 0., 0.],
                                      [s, c, 0., 0.],
                                      [0., 0., 1., 0.],
                                      [0., 0., 0., 1.]])
                rot_mat = np.matmul(rot_mat, rot_mat_z)
            self.counter += 1

        # Transform rotation to NAO coordinate system
        # syzx szxy rzxz

        # 'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        # 'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        # 'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        # 'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        # 'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        # 'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),

        # 'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        # 'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
        rx, ry, rz = euler_from_matrix(rot_mat, axes='szxy')

        cmu_rot_x_name = '{}.Xrotation'.format(root.name)
        cmu_rot_y_name = '{}.Yrotation'.format(root.name)
        cmu_rot_z_name = '{}.Zrotation'.format(root.name)

        if cmu_rot_x_name in CMU_TO_NAO_JOINT_MAPPING:
            nao_joint_name = CMU_TO_NAO_JOINT_MAPPING[cmu_rot_x_name]
            gesture_dict[nao_joint_name] = rx

        if cmu_rot_y_name in CMU_TO_NAO_JOINT_MAPPING:
            nao_joint_name = CMU_TO_NAO_JOINT_MAPPING[cmu_rot_y_name]
            gesture_dict[nao_joint_name] = ry

        if cmu_rot_z_name in CMU_TO_NAO_JOINT_MAPPING:
            nao_joint_name = CMU_TO_NAO_JOINT_MAPPING[cmu_rot_z_name]
            gesture_dict[nao_joint_name] = rz

        for child in root.children:
            gesture_dict = self.extractRootJoint(child, gesture_dict=gesture_dict)

        return gesture_dict

    def toNAOJoint(self, fetch_every=1):
        gesture_list = []
        print (self.num_motions)
        for ind in range(0, self.num_motions, fetch_every):
            self.counter = 0
            self.this_motion = self.all_motions[ind]
            gesture_dict = {key: 0.0 for key in NAO_TO_CMU_JOINT_MAPPING.keys()}
            gesture_dict = self.extractRootJoint(self.root, gesture_dict=gesture_dict)
            gesture_list.append(gesture_dict)
        return gesture_list


def read_tsv_file(file_in):
    sec = []
    word = []
    with open(file_in, 'r') as f:
        data = f.readlines()
        col = 0
        for line in data:
            col = col + 1
            numbers = line.split()
            # sec_word.append({numbers[0],numbers[2]})
            sec.append(numbers[0])
            word.append(numbers[2])
    f.close()
    print len(sec), len(word)
    return sec, word


def NAOInitialization(IP, PORT):
    tts.setLanguage("English")
    tts.say("Hello, world! I am NAO!")


if __name__ == "__main__":
    # IP = "nao.local."
    # PORT = 9559
    # IP = "192.168.1.53"
    # PORT = 9559
    IP = "laptop-8iu8e25s.local."
    PORT = 49964
    motionProxy = ALProxy("ALMotion", IP, PORT)
    speakProxy = ALProxy("ALTextToSpeech", IP, PORT)
    postureProxy = ALProxy("ALRobotPosture", IP, PORT)
    tts = ALProxy("ALTextToSpeech", IP, PORT)
    tts.setLanguage("English")
    tts.say("Hello, world! I am NAO!")

    # convert .bvh file to joint angle

    number = "023"  # 007 008 012 013 011
    # 016 020 023 030 035 036 039
    bvh_file = "G:/Co-Speech_Gesture_Generation-master/my_inference/bvh/bvh_tst_for_final_model/tst_2022_v1_" + number + "_lixiangqi_generated.bvh"
    # bvh_file = "G:/Co-Speech_Gesture_Generation-master/mnt/val/val/bvh/bvh/val_2022_v1_" + number + ".bvh"
    # file_tsv = "G:/Co-Speech_Gesture_Generation-master/mnt/val/val/tsv/tsv/val_2022_v1_" + number + ".tsv"
    file_tsv = "G:/Co-Speech_Gesture_Generation-master/mnt/tst/tst/tsv/tsv/tst_2022_v1_" + number + ".tsv"


    # read in tsv files for speech synchornization
    bvh_test = BVHLoader(bvh_file)
    gesture_list = bvh_test.toNAOJoint()
    df = pd.DataFrame(gesture_list)
    columns = df.columns.tolist()
    print(columns)
    times = len(df["RElbowRoll"].tolist())

    sec, word = read_tsv_file(file_tsv)

    print sec
    # sentence = []
    size = len(sec)
    sentence = [" " for _ in range(size)]
    start_word_time = [-1 for _ in range(size)]
    start_word_time[0] = "0"
    index = 0
    string = " "

    for i in range(1650, len(word)):
        # print seg
        string = string + word[i] + "  "
        print string
        if string.count(", ") or string.count(". ") or string.count("? "):
            # string = " "
            sentence[index] = string
            index = index + 1
            string = " "
            if i + 1 < len(sec):
                start_word_time[index] = sec[i + 1]
            print "detect end of sentense!"

    print sentence
    print start_word_time
    word_count = 0
    # f = open(
    #     'E:/Co-Speech_Gesture_Generation-master/Co-Speech_Gesture_Generation-master/output/robot_joints_data.txt',
    #     'w')

    time_start_1 = time.time()
    old_time = -1
    syn_word = 1

    #### SET VALUE OF synchronization_delay FOR PARTIAL NON-SYNCHRONIZATION
    synchronization_delay = 0  # 0  30  60  90  120  150 / 0, 30, 60, 90, 120, 150
    for i in range(-synchronization_delay, times, 1):

    # for i in range(0, times, 1):
        ### word with gesture --- consistent but not coherent
        # print (i)
        # if word_count >= len(sec):
        #     word_count = len(sec) - 1
        # word_time = sec[word_count]
        # # print word_time
        # word_time_frame = int(float(word_time) * 30)  # 30 frames per seconds
        # if word_time_frame == old_time:
        #     word_time_frame = word_time_frame + 1
        # word_cur = word[word_count].lower()
        # # print i, word_time_frame, word_cur
        # flag = False
        # if word_time_frame == i:
        #     tts.say(word_cur)
        #     # flag[i] = 0
        #     flag = True
        #     old_time = word_time_frame
        #     word_count = word_count + 1
        #     # if word_count >= times:
        #     #     word_count = times

        ### sentence with gesture --- coherent but inconsisitent if blocking
        # print (i)
        if word_count >= len(start_word_time):
            word_count = len(start_word_time) - 1
        word_time = start_word_time[word_count]
        # print word_time
        word_time_frame = int(float(word_time) * 30)  # 30 frames per seconds
        if word_time_frame == old_time:
            word_time_frame = word_time_frame + 1
        word_cur = sentence[word_count]
        flag = False

        if int(float(start_word_time[syn_word]) * 30) == i:
            print "Syn:     " + str(i) + " " + str(int(float(start_word_time[syn_word]) * 30)) + str(sentence[syn_word])
            syn_word = syn_word + 1

        if word_time_frame - synchronization_delay == i:
            # print i, word_time_frame + synchronization_delay, word_cur
            print "NON-Syn: " + str(i) + " " + str(word_time_frame - synchronization_delay) + str(word_cur)
            # tts.post.say(word_cur)

            ## UNCOMMENT FOR COMPLETELY NON-SYNCHRONIZATION
            # tts.say(word_cur)

            flag = True

            old_time = word_time_frame
            word_count = word_count + 1

        ### gesture
        # if i >= 900:(006)
        if i >= 0:  # 002
            for column in columns:
                value_write = 0
                """
                NAO including time sleep
                """
                if column == "LShoulderPitch" or column == "RShoulderPitch":
                    cur_val = (df[column][i] * almath.TO_DEG + 90)
                    if cur_val > 119:
                        cur_val = 119
                    elif cur_val < -119:
                        cur_val = -119
                    motionProxy.post.setAngles(column, cur_val * almath.TO_RAD, 1)
                elif column == "LElbowRoll" or column == "RElbowRoll":
                    cur_val = (-1)*(df[column][i] * almath.TO_DEG)
                    # print cur_val
                    motionProxy.post.setAngles(column, cur_val * almath.TO_RAD, 1)

                else:
                    motionProxy.post.setAngles(column, df[column][i], 1)

        if not flag:
            # time.sleep(0.02)
            time.sleep(0.01)
    # f.close()
    print sentence
    time_end_1 = time.time()
    print(str(time_end_1 - time_start_1) + " seconds")
