# -*- coding: utf-8 -*-#
"""
transfer COCO 17 keypoints to mpii 16 keypoints

Created on 2019/8/14

@author: Qi Zhen
@mail : qizhen816@163.com
@github : https://github.com/qizhen816

"""
import numpy as np

def COCOtoMPII(scores,joints,posescore):

    '''
    scores: 每个关节点的置信度 n*17 n为检测出来的人数
    joints: 关节点坐标 n*17*2
    posescore: 姿态置信度 n

    如果没有姿态置信度，可以传入一个长为n，值为1的列表
    如果没有关节点置信度，可以将检测出来的关节点置信度设为1，没检测出来的为0

    COCO:["nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]
    MPII:["rightAnkle","rightKnee","rightHip",
              "leftHip","leftKnee","leftAnkle",
              "pelvis", "throax","upper_neck", "head_top",
              "rightWrist","rightElbow","rightShoulder",
              "leftShoulder","leftElbow","leftWrist"]
    '''
    new_joints = np.zeros([10,16,2])
    new_scores = np.zeros([10,16])
    def addnew(j,s,i):
        new_joints[:,i]=j
        new_scores[:,i]=s

    def usefulscore(i,j):
        return scores[i][j]>0.1

    def dis(x, y):
        # distance = np.dot((x - y), (x - y)) ** 0.5
        # return np.asarray(distance,-1*distance)
        return np.asarray([(y[0]-x[0]),1*(y[1]-x[1])])
    addnew(joints[:, 16], scores[:, 16],0)  #r_ankle
    addnew(joints[:, 14], scores[:, 14],1)  #r_knee
    addnew(joints[:, 12], scores[:, 12],2)  #r_hip
    addnew(joints[:, 11], scores[:, 11],3)  #l_hip
    addnew(joints[:, 13], scores[:, 13],4)  #l_knee
    addnew(joints[:, 15], scores[:, 15],5)  #l_ankle

    ptuh_j = np.zeros([len(posescore),4,2])
    ptuh_s = np.zeros([len(posescore),4])
    for i,s in enumerate(posescore):
        if s < 0.1:
            # pelvis
            # throax
            # upper_neck
            # head_top
            continue
        if usefulscore(i,12) and usefulscore(i,11):
            pelvis_score = (scores[i,12]+scores[i,11])/2
            x = scores[i,12]
            y = scores[i,11]
            distance = np.dot((x - y), (x - y)) ** 0.5
            pelvis_cood = (joints[i,12]+joints[i,11])/2 + np.asarray([-0.3*distance,0])
        elif usefulscore(i,12):
            pelvis_score = scores[i,12]
            pelvis_cood = joints[i,12]
        elif usefulscore(i,11):
            pelvis_score = scores[i,11]
            pelvis_cood = joints[i,11]
        else:
            pelvis_score = 0
            pelvis_cood = np.asarray([0,0])
        pelvis_cood = pelvis_cood.astype(np.int)
        pelvis_cood = np.fmax(pelvis_cood,np.asarray([0,0]))
        # pelvis_cood = np.fmin(pelvis_cood,np.asarray([337,337]))
        ptuh_s[i,0] = pelvis_score
        ptuh_j[i,0] = pelvis_cood


        # 鼻子作为脑袋的中心点 尺度距离=鼻子到眼睛中心点
        # 头顶是鼻子+尺度距离*3
        # 脖子是鼻子-尺度距离*3
        # 胸口是鼻子-尺度距离*6
        if usefulscore(i,0):
            c = 0
            center_score = scores[i,0]
            center_cood = joints[i,0]
        elif usefulscore(i,3) and usefulscore(i,4):
            #没有鼻子就是耳朵中心作为中心点
            c = 1
            center_score = (scores[i,3]+scores[i,4])/2
            center_cood = (joints[i,3]+joints[i,4])/2
        elif usefulscore(i,1) and usefulscore(i,2):
            #没有耳朵就是眼睛中心作为中心点
            c = 2
            center_score = (scores[i,1]+scores[i,2])/2
            center_cood = (joints[i,1]+joints[i,2])/2
        elif usefulscore(i,1):
            #再不行就单个眼睛
            c = 2
            center_score = scores[i,1]
            center_cood = joints[i,1]
        elif usefulscore(i,2):
            c = 2
            center_score = scores[i,2]
            center_cood = joints[i,2]
        else:
            c = 3
            center_score = 0
            center_cood = np.asarray([0, 0])

        if usefulscore(i,1) and usefulscore(i,2):
            other_cood = (joints[i,1]+joints[i,2])/2
            tmp_score = (center_score + scores[i,1] + scores[i,2]) / 3
        elif usefulscore(i,1):
            other_cood = joints[i,1]
            other_cood[0] = center_cood[0]
            tmp_score = (center_score + scores[i,1]) / 2
        elif usefulscore(i,2):
            other_cood = joints[i,2]
            other_cood[0] = center_cood[0]
            tmp_score = (center_score + scores[i,2]) / 2
        else:
            other_cood = np.asarray([0, 0])
            tmp_score = 0

        if (c == 0 or c ==1) and tmp_score!=0:
            # 鼻子/耳朵为中心点 且 眼睛存在
            tmp_score = center_score
            distance = dis(center_cood,other_cood)
            throax_cood = center_cood - distance*6
            neck_cood = center_cood - distance*3
            head_cood = center_cood + distance*3
        elif c != 3:
            # 鼻子/耳朵为中心点 但 眼睛不存在
            # 眼睛作为中心点
            # 中心点 尺度距离=两个肩膀中心到眼睛中心点
            # 头顶是眼睛+尺度距离
            # 脖子是鼻子-尺度距离*0.5
            # 胸口是眼睛-尺度距离*1.5
            tmp_score = center_score
            if usefulscore(i,5) and usefulscore(i,6):
                    distance = dis(center_cood, (joints[i,5] + joints[i,6]) / 2)
                    throax_cood = center_cood - distance * 1.5
                    neck_cood = center_cood - distance * 0.5
                    head_cood = center_cood + distance
            elif usefulscore(i,5):
                    distance = dis(center_cood, joints[i,5])*0.75
                    throax_cood = center_cood - distance * 1.5
                    neck_cood = center_cood - distance * 0.5
                    head_cood = center_cood + distance
            elif usefulscore(i,6):
                    distance = dis(center_cood, joints[i,6])*0.75
                    throax_cood = center_cood - distance * 1.5
                    neck_cood = center_cood - distance * 0.5
                    head_cood = center_cood + distance
            else:
                    throax_cood = center_cood
                    neck_cood = center_cood
                    head_cood = center_cood
        else:
            tmp_score = 0
            throax_cood = np.asarray([0, 0])
            neck_cood = np.asarray([0, 0])
            head_cood = np.asarray([0, 0])

        thorax_score = tmp_score
        if usefulscore(i,5) and usefulscore(i,6) and tmp_score!=0:
            throax_cood[1] = (joints[i,5,1]+joints[i,6,1])/2
            neck_cood = (center_cood + (joints[i, 5] + joints[i, 6]) / 2) / 2
        elif tmp_score == 0:
            throax_cood = (joints[i,5]+joints[i,6])/2
            thorax_score = (scores[i,6]+scores[i,5])/2
            neck_cood = (center_cood + (joints[i, 5] + joints[i, 6]) / 2) / 2


        throax_cood = np.fmax(throax_cood.astype(np.int),np.asarray([0,0]))
        # throax_cood = np.fmin(throax_cood.astype(np.int),np.asarray([22,22]))
        neck_cood = np.fmax(neck_cood.astype(np.int),np.asarray([0,0]))
        # neck_cood = np.fmin(neck_cood,np.asarray([337,337]))
        head_cood = np.fmax(head_cood.astype(np.int),np.asarray([0,0]))
        # head_cood = np.fmin(head_cood,np.asarray([337,337]))
        ptuh_s[i,1] = thorax_score
        ptuh_j[i,1] = throax_cood
        ptuh_s[i,2] = tmp_score
        ptuh_j[i,2] = neck_cood
        ptuh_s[i,3] = tmp_score
        ptuh_j[i,3] = head_cood

    addnew(ptuh_j[:,0],ptuh_s[:,0],6) #pelvis
    addnew(ptuh_j[:,1],ptuh_s[:,1],7) #throax
    addnew(ptuh_j[:,2],ptuh_s[:,2],8) #upper_neck
    addnew(ptuh_j[:,3],ptuh_s[:,3],9) #upper_neck

    addnew(joints[:, 10], scores[:, 10],10)  #r_wrist
    addnew(joints[:, 8], scores[:, 8],11)  #r_elbow
    addnew(joints[:, 6], scores[:, 6],12)  #r_shoulder
    addnew(joints[:, 5], scores[:, 5],13)  #l_shoulder
    addnew(joints[:, 7], scores[:, 7],14)  #l_elbow
    addnew(joints[:, 9], scores[:, 9],15)  #l_wrist

    new_joints = fix_pose(new_joints,new_scores,posescore=posescore)

    return new_joints

def fix_pose(joints,jointscores,posescore=[],minscore = 0.1):
    # joints:n,16,2 jointscores:n,16 posescore:n
    # 该函数是为了避免显示时失真，在关节点丢失时使用父亲节点的坐标对其赋值
    joints_new = []
    parents = [1,2,6,6,3,4,7,8,9,-1,11,12,7,7,13,14]
    for i in range(len(jointscores)):
        if len(posescore)!=0:
            if posescore[i] < minscore:
                continue
        tmp_joint = joints[i]
        for k in range(16):
            if jointscores[i][k]<minscore:
                x = parents[k]
                while x!=-1:
                    if jointscores[i][x]>minscore:
                        tmp_joint[k] = tmp_joint[x]
                        break
                    x = parents[x]
        joints_new.append(tmp_joint)
    if joints_new == []:
        joints_new.append(np.zeros([16,2]))
    print(len(joints_new),joints_new[0].shape)
    return np.asarray(joints_new)