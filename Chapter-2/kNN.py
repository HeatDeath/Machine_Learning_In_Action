# -*- coding:utf-8 -*-
__author__ = 'hotdeath'
"""
k-邻近算法，采用测量不同特征值之间的距离方法进行分类

优点：
    精度高，对异常值不敏感，无数据输入假定

缺点：
    计算复杂度高，空间复杂度高

使用数据范围：
    数值型 和 标称型

工作原理：
    存在一个样本数据集合，也称作训练样本集，并且样本集中的每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。
输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应特征进行比较，然后算法提取样本集中特征最相似数据（最邻近）的分类
标签。

一般流程：
（1）收集数据： 可以使用任何方法
（2）准备数据： 距离计算所需要的数值，最好是结构化的数据格式
（3）分析数据： 可以使用任何方法
（4）训练算法： 此步骤不使用于 k-邻近算法
（5）测试算法： 计算错误率
（6）使用算法： 首先需要输入样本数据和结构化的输出结果，然后运行 k-邻近算法 判定输入数据分别属于哪个分类，最后应用对计算出的
分类执行后续的处理。
"""
from numpy import *
import operator


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# group, labels = create_data_set()
# print(group)
# print(type(group))
# print(labels)


def classify0(inX, dataSet, labels, k):
    """
    分类器 v1.0
    :param inX: 用于分类的输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 标签向量（标签向量的元素数目和矩阵 dataset 的行数相同）
    :param k: 用于选择最近邻居的数目
    :return: 排序首位的 label

    对未知类别属性的数据集中的每个点依次执行以下操作：
    1、计算已知类别数据集中的点与当前点之间的距离
    2、按照距离递增次序排序
    3、选取与当前点距离最小的 k 个点
    4、确定前 k 个点所在类别的出现频率
    5、返回前 k 个点出现频率最高的类别作为当前点的预测分类
    """
    # ndarray.shape 数组维度的元组，ndarray.shape[0]表示数组行数，ndarray.shape[1]表示列数
    dataSetSize = dataSet.shape[0]
    # print(dataSetSize)

    # 将输入的 inX（1*2） 进行扩展，扩展为 4*2 矩阵，使其与训练样本集中的数据（4*2）矩阵作减法
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # print(diffMat)

    # 将 差值矩阵 的每一项乘方
    sqDiffMat = diffMat**2
    # print(sqDiffMat)

    # 在指定的轴向上求得数组元素的和
    sqDistances = sqDiffMat.sum(axis=1)
    # print(sqDistances)

    # 开方
    distances = sqDistances**0.5
    # print(distances)

    # 将 distances 数组的元素排序 返回由其索引组成的 list
    sortedDistIndicies = distances.argsort()
    # print(sortedDistIndicies)

    # classCount 字典用于类别统计
    classCount = {}

    # 遍历 sortedDistIndicies list，依次获取最近的 k 个邻居对应的 label
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # print(voteIlabel)

        # 若 classCount 字典中不存在 当前 voteIlabel ，则置该 key voteIlabel 对应的 value 为 0
        # 否则 +1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # print(classCount)

    # print(classCount)

    # 将 classCount 字典进行排序，按照 items 的值，倒序（从大到小排列）
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedClassCount)

    # 将排序首位的 label 作为返回值
    return sortedClassCount[0][0]

# print(classify0([0, 0], group, labels, 3))

"""
k-近邻算法是基于实例的学习，使用算法时我们必须有接近实际数据的训练样本数据
k-近邻算法必须保存全部的数据集，如果训练数据集很大，必须使用非常大的存储空间
此外，由于必须对数据集中的每个数据计算距离值，实际使用时可能会非常耗时
另一个缺陷是，它无法给出任何数据的基础结构信息，因此我们也无法知晓平均实例样本和典型实例样本具有什么特征
"""






