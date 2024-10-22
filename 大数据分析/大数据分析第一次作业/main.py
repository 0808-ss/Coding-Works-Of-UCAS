# coding=utf-8
import numpy as np
# import pandas as pd
import pandas as pd
import scipy as sp
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import chardet




def GetData(txt_name):
    user_id = []
    rest_id = []
    data = []
    with open(txt_name) as file:
        for line in file.readlines():
            temp = line.strip().split(' ')
            user_id.append(float(temp[0]))
            rest_id.append(float(temp[1]))
            data.append(float(temp[2]))
    return user_id,rest_id,data

def GetData_txt(txt_name):
    user_id = []
    rest_id = []
    data = []
    with open(txt_name, 'r') as file:
        for line in file:
            # 去除行尾的换行符并分割行
            temp = line.strip().split(' ')
            # 检查分割后的列表是否有足够的元素
            if len(temp) == 3 and temp[0] and temp[1] and temp[2]:
                user_id.append(int(temp[0]))  # 将 user_id 转换为整数
                rest_id.append(int(temp[1]))  # 将 rest_id 转换为整数
                data.append(float(temp[2]))  # data 可以是浮点数
            else:
                print(f"Skipping invalid line: {line.strip()}")
    return user_id, rest_id, data

def DrawFigure(UV):
    for i in range(0, 9):  # 一共9张图
        print(i)
        # 取出数据
        x = UV[:, i + 1]
        y = UV[:, i]

        # label = 'u' + str(i + 1) + '-->' + 'u' + str(i + 2)

        label = 'v' + str(i + 1) + '-->' + 'v' + str(i + 2)
        plt.scatter(x, y, s=0.5, marker='.', color='b', label=label)
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))

        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))

        # plt.title("Spectral Plot of u%d and u%d" % (i + 1, i + 2))

        plt.title("Spectral Plot of v%d and v%d" % (i + 1, i + 2))

        font = {'family': 'FangSong',
                'weight': 'normal',
                'size': 20,
                }

        # plt.xlabel("u%d" % (i + 2), font)
        # plt.ylabel("u%d" % (i + 1), font)

        plt.xlabel("v%d" % (i + 2),font)
        plt.ylabel("v%d" % (i + 1),font)

        plt.legend(fontsize=15)
        plt.show()

if __name__ == '__main__':
    #初始化数据
    user_id = []
    rest_id = []
    data = []

    # user_id,rest_id,data=GetData('yelp.edgelist')
    # 根据行和列构造稀疏矩阵
    user_id,rest_id,data=GetData_txt('ratings_data.txt')
    sm=sp.sparse.coo_matrix((data,(user_id,rest_id)),shape = (len(user_id),len(rest_id)))
    # 奇异值分解,维度选择为10
    U,S,V = svds(sm, 10)
    #画图
    DrawFigure(U)
    # DrawFigure(np.transpose(V))

# 部分代码参考自:https://blog.csdn.net/weixin_42200347/article/details/124742769
# 作业也参考自:https://blog.csdn.net/weixin_42200347/article/details/124742769
