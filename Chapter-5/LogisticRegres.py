"""
假设现在有一些数据点，我们用一条直线对这些点进行拟合（该直线成为最佳拟合直线），这个拟合的过程称为回归

主要思想：
    根据现有数据对分类边界线建立回归公式

训练分类器时的做法就是训在最佳拟合参数，使用的是最优化算法

------------------------------------------

Logistic 回归的一般过程

1、收集数据：
    采用任一方法收集数据
2、准备数据：
    由于需要进行距离计算，因此要求数据类型为数值型。结构化数据格式最佳
3、分析数据：
    采用任一方法对数据进行分析
4、训练算法：
    大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数
5、测试算法：
    一旦训练步骤完成，分类将会很快完成
6、使用算法：
    首先，需要输入一些数据，并将其装换成对应的结构化数值
    接着，基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定其属于哪个类别
    最后，在输出的类别上做一些其他分析工作
"""
"""
Logistic 回归

1、优点：
    计算代价不高，易于理解和实现
2、缺点：
    容易欠拟合，分类精度可能不高
3、适用数据类型：
    数值型和标称型数据

函数输出 0 或 1，具有这种性质的函数成为 海维赛德阶跃函数（Heaviside step function），或直接成为 单位阶跃函数
该函数在跳跃点上从0瞬间跳跃到1，这个瞬间跳跃过程有时候很难处理，另一个函数性质相似， Sigmoid 函数

Sigmoid 函数：
    当 x 为 0 时，Sigmoid 函数值为 0.5，
    随着 x 的增大，对应的 Sigmoid 值将逼近于 1 ；
    随着 x 的减小，对应的 Sigmoid 值将逼近于 0.
    
    为了实现 Logistic 回归分类器，在每个特征上都乘以一个回归系数，然后把所有的结果值相加，将这个总和带入 Sigmoid 函数中，
进而得到一个范围在 0-1 之间的数值。

Logistic 回归 也可以被看做是一种 概率估计

"""
"""
梯度上升法：

1、思想：
    要找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻

梯度上升算法到达每个点后都会重新估计移动方向，循环迭代，直到满足停止条件

梯度算子，总是指向函数值增长最快的方向

移动量 的 大小，称为 步长

"""
import numpy as np

def loadDataSet():
    """
    打开文本文件，并逐行读取
    :return:
    """
    # 数据
    dataMat = []

    # 标签
    labelMat = []

    # 打开测试集文本
    fr = open('testSet.txt')

    # 逐行读取
    for line in fr.readlines():
        # 去掉 去读行 首尾 空格 ，并 以 空白 进行拆分
        lineArr = line.strip().split()

        # X0 X1 X2
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])

        # 标签
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    """
    sigmoid 函数
    :param inX:
    :return:
    """
    return 1.0/(1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """

    :param dataMatIn:
    :param classLabels:
    :return:
    """
    # 将 数据集 转换为 numpy 中的 matrix
    dataMatrix = np.mat(dataMatIn)

    # 将 分类标签 集 转换为 numpy 矩阵，并进行 转置（从 行向量 转换为 列向量）
    labelMat = np.mat(classLabels).transpose()

    # 获得 数据矩阵 的 形状
    m, n = np.shape(dataMatrix)

    # 向 目标 移动的步长
    alpha = 0.001

    # 迭代次数
    maxCycles = 500

    # 回归系数 矩阵（列向量）
    weights = np.ones((n, 1))

    for k in range(maxCycles):

        # h 为 列向量，列向量的元素个数为 样本个数
        h = sigmoid(dataMatrix*weights)

        # 计算 真是类别 与 预测类别 的 差值
        error = (labelMat - h)

        # 按照 差值 方向 调整 回归系数
        weights = weights + alpha*dataMatrix.transpose()*error

    return weights

# result_weight = gradAscent(loadDataSet()[0], loadDataSet()[1])
# print(result_weight)
# print(type(result_weight))


def plotBestFit(weights):
    import matplotlib.pyplot as plt

    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)

    n = np.shape(dataArr)[0]

    xcord1 = []
    ycord1 = []

    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        if int(labelMat[i]) == 0:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# plotBestFit(result_weight.getA())

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    import random

    dataMatrix = np.array(dataMatrix)

    m, n = np.shape(dataMatrix)

    # 行向量
    weights = np.ones(n)

    for j in range(numIter):
        dataIndex = list(range(m))

        for i in range(m):

            # alpha 在每次迭代的时候都会调整，缓解数据波动或者高频波动
            # 随着迭代次数不断减小，但永远不会减小到 0，以此保证新数据仍然具有一定影响
            # 如果处理的问题是动态变化的，那么可以适当加大常数项，以此来确保新的值获得更大的回归系数
            # j 是 迭代次数， i 是 样本点的下标
            alpha = 4/(1.0 + j + i) + 0.01

            # 从 数据集 中 随机选择样本
            randIndex = int(random.uniform(0, len(dataIndex)))

            # 数值
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))

            # 数值
            error = classLabels[dataIndex[randIndex]] - h

            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]

            del(dataIndex[randIndex])

    return weights


# result_weight = stocGradAscent1(loadDataSet()[0], loadDataSet()[1])
# print(result_weight)
# plotBestFit(result_weight)

