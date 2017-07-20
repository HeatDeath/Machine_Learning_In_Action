"""
在约会网站上使用 kNN
1.收集数据： 提供文本文件
2.准备数据： 使用 Python 解析文本文件
3.分析数据： 使用 Matplotlib 画二维扩散图
4.训练算法： 此步骤不适合 Kk-近邻算法
5.测试算法：
    测试样本与非测试样本的区别在于：
        测试样本是已经完成分类的数据，如果预测分类与实际类别不用，则标记为一个错误
6.使用算法： 产生简单的命令行程序，然后可以输入一些特征数据以判断对方是否为自己喜欢的类型
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from kNN import classify0


def file2matrix(filename):
    """
    将读取的文件转换为矩阵
    :param filename: 文件名
    :return: 转换后的矩阵
    """
    # 打开文件
    fr = open(filename)

    # 将文件内容按行读取为一个 list
    arrayOLines = fr.readlines()

    # 获取 list 的长度，即文件内容的行数
    numberOfLines = len(arrayOLines)

    # 生成一个 numberOfLines*3 并以 0 ，进行填充的矩阵
    returnMat = np.zeros((numberOfLines, 3))

    # 分类标签 向量
    classLabelVector = []

    #
    index = 0

    # 遍历读入文件的每一行
    for line in arrayOLines:
        # 截取掉所有的回车符
        line = line.strip()

        # 将 line 以空格符进行分割
        listFromLine = line.split('\t')

        # index 行的所有元素替换为 listFromLine 中的 [0:3]
        returnMat[index,:] = listFromLine[0:3]

        # 分类标签向量 list 中添加 listFromLine 中的最后一项
        classLabelVector.append(int(listFromLine[-1]))

        #
        index += 1
    return returnMat, classLabelVector

datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')


def get_figure(datingDataMat, datingLabels):
    """
    直接浏览文本文件方法非常不友好，一般会采用图形化的方式直观地展示数据
    :param datingDataMat:
    :param datingLabels:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 使用 datingDataMat 矩阵的第二、第三列数据
    # 分别表示特征值“玩视频游戏所消耗时间百分比”和“每周所消费的冰淇淋公升数”
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])

    # 利用变量 datingLabels 存储的类标签属性，在散点图上绘制色彩不等，尺寸不同的点
    # scatter plot 散点图
    ax.scatter(datingDataMat[:,0], datingDataMat[:,1],
               15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    plt.show()

# get_figure(datingDataMat, datingLabels)


def autoNorm(dataSet):
    """
    方程中数字差值最大的属性对计算结果的影响最大，在处理这种不同范围的特征值时，采用将数值归一化的方法
    :param dataSet: 输入的数据集
    :return: 归一化后的数据集
    """
    # dataSet.min(0) 中的参数 0 使得函数可以从列中选取最小值，而不是选当前行的最小值
    # minVals 储存每列中的最小值
    minVals = dataSet.min(0)

    # maxVals 储存每行中的最小值
    maxVals = dataSet.max(0)

    # 求得差值
    ranges = maxVals - minVals

    #
    # normDataSet = np.zeros(np.shape(dataSet))

    # 将数据集 dataSet 的行数放入 m
    m = dataSet.shape[0]

    # 归一化
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals

# normDataSet, ranges, minVals = autoNorm(datingDataMat)

def datingClassTest():

    # 选择 10% 的数据作为测试数据，90% 的数据为训练数据
    hoRatio = 0.10

    # 将输入的文件转换为 矩阵形式
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    # 将特征值归一化
    normDataSet, ranges, minVals = autoNorm(datingDataMat)

    # 计算测试向量的数量
    m = normDataSet.shape[0]
    numTestVecs = int(m*hoRatio)

    # 错误数量统计
    errorCount = 0.0

    # 遍历 测试向量
    for i in range(numTestVecs):

        # # 取 数据集 的后 10% 作为测试数据，错误率为 5%

        # 调用 classify0() 函数
        # 以归一化后的的数据集 normDataSet 的第 i 行数据作为测试数据，
        # 以 numTestVecs:m 行数据作为训练数据，
        # datingLabels[numTestVecs:m] 作为标签向量，
        # 选择最近的 3 个邻居
        classifierResult = classify0(normDataSet[i,:], normDataSet[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m], 3)

        # 打印 预测结果 与 实际结果
        print("the classifier came back with: %d,"
              "the real answer is: %d " % (classifierResult, datingLabels[i]))

        # 当预测失败时，错误数量 errorCount += 1
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

        # # -----------------------------------------------------------------------
        # # 取 数据集 的后 10% 作为测试数据，错误率为 6%
        # classifierResult = classify0(normDataSet[m-numTestVecs+i, :], normDataSet[:m-numTestVecs, :],
        #                              datingLabels[:m-numTestVecs], 3)
        #
        # print("the classifier came back with: %d,"
        #       "the real answer is: %d " % (classifierResult, datingLabels[m-numTestVecs+i]))
        #
        # if classifierResult != datingLabels[m-numTestVecs+i]:
        #     errorCount += 1.0
        # # -----------------------------------------------------------------------


    print("the total error rate is : %f" % (errorCount/float(numTestVecs)))

# datingClassTest()

def classifyPerson():
    # 预测结果 list
    resultList = ['not at all', 'in small doses', 'in large doess']

    # 获取用户输入
    percentTats = float(input('percentage of time spent playing video games?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    ffMile = float(input('frequent flier miles earned per year?'))

    # 归一化数据集
    normDataSet, ranges, minVals = autoNorm(datingDataMat)

    # 将用户输入转化为一个 Matrix
    inArr = np.array([ffMile, percentTats, iceCream])

    # 调用 classify0() ，将用户输入矩阵归一化后进行运算
    classifierResult = classify0((inArr - minVals)/ranges, normDataSet, datingLabels, 3)

    # 打印预测结果
    print('You will probably like this person: ', resultList[classifierResult])

classifyPerson()