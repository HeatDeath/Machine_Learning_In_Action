"""
使用 Logistic 回归估计 马伤病死亡率

1、收集数据：
    给定数据文件

2、准备数据：
    用 Python 解析文本并填充缺失值

3、分析数据：
    可视化，并观察数据

4、训练算法：
    使用优化算法，找到最佳的系数

5、测试算法：
    为了量化回归的效果，需要观察错误率。
    根据错误率决定是否回退到训练阶段，通过改变迭代的次数和步长等参数得到更好的回归系数

6、使用算法：
    实现一个简单的命令行程序来收集马的症状并输出预测结果很容易

说明：
    数据集中有 30% 的数据时缺失的

解决数据缺失的方法：

1、使用可用特征的均值来填补缺失值

2、使用特殊值来填补缺失值

3、忽略有缺失值的样本

4、使用相似样本的均值来填补缺失值

5、使用其他的机器学习算法来预测缺失值

"""
from LogisticRegres import *

def classifyVector(inX, weights):
    """

    :param inX: 特征向量
    :param weights: 回归系数
    :return: 分类结果
    """
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')

    traingingSet = []
    traingingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split()
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        traingingSet.append(lineArr)
        traingingLabels.append(float(currLine[21]))

    traingingWeights = stocGradAscent1(np.array(traingingSet), traingingLabels, 500)

    errorCount = 0

    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split()
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))

        if int(classifyVector(np.array(lineArr), traingingWeights)) != int(currLine[21]):
            errorCount += 1

    errorRate = float(errorCount)/numTestVec

    print("the error rate is %f" % errorRate)

    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after {} iterations the average error rate is : {}".format(numTests, errorSum/float(numTests)))

multiTest()

"""
Logistic 回归的目的是寻找一个非线性函数 Sigmoid 的最佳拟合参数，求解过程可以由最优化算法来完成

随机梯度上升算法 与 梯度上升算法 效果相当，但占用更少的计算资源

随机梯度上升算法是一个在线算法，可以在新数据到来时完成参数更新，而不需要重新读取整个数据集来进行运算
"""