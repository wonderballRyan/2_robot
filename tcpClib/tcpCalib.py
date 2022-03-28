import util
import os
import scipy as sp

dirPath = os.path.dirname(os.path.abspath("_file_"))
posePath = os.path.join(dirPath, "pose.txt")

pose = []
with open(posePath, "r") as f:
    for line in f.readlines():
        temp = []
        line = line.strip("\n")
        line = line.split(" ")
        for i in range(len(line)):
            temp.append(float(line[i]))
        pose.append(temp)

poseMatrix = []
for i in range(len(pose)):
    poseMatrix.append(util.qt2rm(pose[i]))
# print(poseMatrix)

A = []
B = []
for i in range(3):
    tempRot = poseMatrix[i][0:3, 0:3] - poseMatrix[i + 1][0:3, 0:3]
    tempTrans = poseMatrix[i + 1][0:3, 3] - poseMatrix[i][0:3, 3]
    for row in tempRot:
        A.append(list(row))
    for i in tempTrans:
        B.append([i])

print(A)
print(B)

# 非方阵矩阵求解
invA = sp.linalg.pinv(A)
X = invA.dot(B)
print(X)
