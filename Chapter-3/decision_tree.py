"""
决策树，可以使用不熟悉的数据集合，并从中提取出一系列规则，在这些机器根据数据集创建规则时，就是是机器学习的过程

优点：
    计算复杂度不高，输出易于理解的结果，对于中间值的缺失不敏感，可以处理不相关特征数据
缺点：
    可能会产生过度匹配问题
适用的数据类型：
    数值型和标称型
一般流程：
    1、收集数据：
            可以使用任何方法
    2、准备数据：
            树构造算法只适用于标称型数据，数值型数据必须离散化
    3、分析数据：
            可以使用任何方法，构造树完成后，应该检查图形是否符合预期
    4、训练算法：
            构造树的数据结构
    5、测试算法：
            使用经验树计算错误率
    6、使用算法：
            此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义
"""

from math import log
import operator
from treePlotter import retrieveTree, createPlot

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

myDat, labels = createDataSet()

# 划分数据集的大原则是： 将无序的数据变得更加有序
# 信息增益（information gain）： 在划分数据集之前之后信息发生的变化
# 获得 信息增益 最高 的特征就是最好的选择
# 集合信息 的 度量方式成为 香农熵（ShannonEntropy） 或 熵（entropy）

def calcShannonEnt(dataSet):
    """
    计算给定数据集的 香农熵
    :param dataSet:
    :return:
    """
    # 计算输入数据集的实例总数
    numEntries = len(dataSet)

    # 创建一个用于统计 label 出现次数的 dict
    labelCounts = {}

    # 遍历数据集中的 特征向量
    for featVec in dataSet:
        # 当前 特征向量 的 标签 为 featVec[-1]
        currentLabel = featVec[-1]

        # 记录标签的出现次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 香农熵 初始化
    shannonEnt = 0.0

    # 遍历字典 labelCounts 的 key
    for key in labelCounts:

        # 计算 label 出现的 频率
        prob = float(labelCounts[key])/numEntries

        # 计算 香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# # 熵越高，则混合的数据越多
# print(calcShannonEnt(myDat))

def splitDataSet(dataSet, axis, value):
    """
    按照给定的特征划分数据集
    遍历数据集中的每一个元素，一旦发现符合要求的值，则将其添加到新创建的列表中
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return: 划分后的数据集
    """
    retDataSet = []

    # 遍历数据集的每一个特征向量
    for featVec in dataSet:

        # 如果特征向量的 axis 项的值与输入的 value 相同
        if featVec[axis] == value:

            # 将 特征向量 featVec 中除了 axis 项之外的其余项保存到 reducedFeatVec list 中
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])

            # 将 reducedFeatVec 追加到 retDataSet 中
            retDataSet.append(reducedFeatVec)
    return retDataSet

# print(myDat)
# print(splitDataSet(myDat, 0, 1))
# print(splitDataSet(myDat, 1, 1))
"""
[[1, 'yes'], [1, 'yes'], [0, 'no']]
[[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]
"""

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    选取特征，划分数据集，计算得出最好的划分数据集的特征
    传入的 dataSet 需要满足：
        1、数据必须是一种由列表元素组成的 list，而且所有的列表元素都要具有相同的数据长度
        2、数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签
    :param dataSet:
    :return:
    """
    # 判定当前数据集包含多少特征属性（列）
    numFeatures = len(dataSet[0]) - 1

    # 计算输入 dataSet 的香农熵
    baseEntropy = calcShannonEnt(dataSet)

    # 初始化 最优 信息增益
    bestInfoGain = 0.0

    # 初始化 最优 特征
    bestFeature = -1

    # 遍历 特征向量 除了 类别标签 外的每一个 特征属性
    for i in range(numFeatures):

        # featureList 存放 当前 列 的 特征属性
        featureList = [example[i] for example in dataSet]

        # 去掉重复值
        uniqueVals = set(featureList)
        print(uniqueVals)

        # 初始化 新的 香农熵
        newEntropy = 0.0

        # 遍历 去掉重复值 后的 特征属性 list（以每一个独一无二的属性值去划分数据集）
        for value in uniqueVals:

            # 划分数据集
            subDataSet = splitDataSet(dataSet, i ,value)

            # 求得 划分后的数据集 占 原始数据集 的 比重
            prob = len(subDataSet)/float(len(dataSet))

            # 将 比重 乘以 划分后的数据集 的 香农熵 并 求和
            newEntropy += prob*calcShannonEnt(subDataSet)

        # 信息增益 等于 基本香农熵 - 当前香农熵
        infoGain = baseEntropy - newEntropy

        # 如果 信息增益 大于 最优 信息增益
        if infoGain > bestInfoGain:

            # 最优信息增益 为 当前 信息增益
            bestInfoGain = infoGain

            # 最优 特征（列） 为 当前 列
            bestFeature = i
    return bestFeature

# chooseBestFeatureToSplit(myDat)

"""
决策树 工作原理：
    得到原始数据集，然后基于最好的属性值划分数据集，由于特征值可能多余两个，因此可能存在大于两个分支的数据集
划分。第一次划分后，数据将被向下传递到树分支的下一个节点，在这个节点上，我们可以再次划分数据。

递归结束的条件：
    程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。如果所有实例具有相同的分类，
则得到一个叶子节点或终止块。任何到达叶子结点的数据必然属于叶子节点的分类。    
"""

def majorityCnt(classList):
    """
        如果数据集已经处理了所有属性，但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子节点，在这种情况下，
    我们通常会采用多数表决的方式。

    返回出现次数最多的分类名称
    :param classList: 类标签 list
    :return: 出现次数最多的 类标签
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """

    :param dataSet: 数据集
    :param labels: 标签列表
    :return:
    """
    # classList 存储数据集的所有类标签
    classList = [example[-1] for example in dataSet]

    # 当 所有的 类标签 完全相同 的时候，直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 当 使用完了 所有特征，仍然 不能 将数据集划分成 仅包含 唯一类别 的分组
    if len(dataSet[0]) == 1:

        # 调用 majority() 返回当前数据集中，出现次数最多的分类名称
        return majorityCnt(classList)

    # 选择数据集中的 最优特征(列)
    bestFeature = chooseBestFeatureToSplit(dataSet)

    # 最优特征（列） 对应的 label
    bestFeatureLabel = labels[bestFeature]

    # 存储树的所有信息
    myTree = {bestFeatureLabel:{}}

    # 从 labels list 中删除 最优特征 对应的 label
    del(labels[bestFeature])

    # 遍历 dataSet 从中提取出 最优特征（列） 的 所有 属性值
    featureValues = [example[bestFeature] for example in dataSet]

    # 去掉重复值
    uniqueValues = set(featureValues)

    # 遍历 独一无二 的属性值
    for value in uniqueValues:

        # 将 剩下的 label 存入 subLabels 中
        subLabels = labels[:]

        # 递归的调用 createTree() ，得到的返回值将被插入 myTree 中
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value),
                                                     subLabels)
    return myTree


# myTree = createTree(myDat, labels)
# print(myTree)

def classify(inputTree, featureLabels, testVec):
    # 根节点
    firstStr = list(inputTree.keys())[0]
    # 子节点
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            # 如果当前节点为 根节点
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featureLabels, testVec)
            # 如果当前节点为 叶子节点
            else:
                classLabel = secondDict[key]
    return classLabel

myTree = retrieveTree(0)
# print(myTree)
# result_1 = classify(myTree, labels, [1, 0])
# result_2 = classify(myTree, labels, [1, 1])
# print(result_1, result_2)

def storeTree(inputTree, fielname):
    import pickle
    fw = open(fielname, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

# storeTree(myTree, 'classifierStorage.txt')
# print(grabTree('classifierStorage.txt'))

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
createPlot(lensesTree)



