"""
朴素贝叶斯

优点：
    在数据较少的情况下仍然有效，可以处理多类别问题
缺点：
    对于输入数据的准备方式比较敏感
使用数据类型：
    标称型数据

贝叶斯决策理论思想核心：
    选择具有最高概率的决策

一般过程：
1、收集数据：
    可以使用任意方法，本章使用 RSS 源

2、准备数据：
    需要数值型或布尔型数据

3、分析数据：
    有大量特征时，绘制特征作用不大，此时使用直方图效果更好

4、训练算法：
    计算不同的独立特征的条件概率

5、测试算法：
    计算错误率

6、使用算法：
    一个常见的朴素贝叶斯应用是文档分类器。可以在任意的分类场景中使用朴素贝叶斯分类器，不一定是文本

由统计学知识可知：
    如果每个特征需要 N 个样本，那么对于 10 个特征将需要 N**10 个样本。如果特征之间相互独立，那么
样本数量可以减少到 1000*N 个。

独立（independence）：
    指统计学意义上的独立，即一个特征或者单词出现的可能性与它和其他单词相邻没有关系

朴素（native）：
    朴素贝叶斯分类器中的朴素即指数据集中的各个特征相互独立

另一个假设：
    每个特征同等重要
"""

import numpy as np

def loadDataSet():
    """
    创建一些试验样本
    """
    # 进行词条切分后的文档集合
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 类别标签集合
    # 0 代表正常文档， 1 代表侮辱性文档
    # 标注信息用于训练程序一遍自动检测侮辱性留言
    classVec = [0, 1, 0, 1, 0, 1]

    return postingList, classVec

# listOPosts, listClasses = loadDataSet()

def createVocabList(dataSet):
    """
    创建一个包含在所有文档中出现的不重复词的 list
    """
    # 创建一个空集
    vocabSet = set([])

    # 创建两个集合的并集
    for document in dataSet:

        # 将每篇文档返回的新词集合添加到该集合中
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document)

    return list(vocabSet)

# myVocabList = createVocabList(listOPosts)
# print(myVocabList)
#
# index_stupid = myVocabList.index('stupid')
# print(index_stupid)

# 词集模型
def setOfWord2Vec(vocabList, inputSet):
    """
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 文档向量，向量的每个元素为 1 或 0，分别表示词汇表中的单词再输入文档中是否出现
    """

    # 创建一个其中所含元素都为 0 的向量，长度与词汇表相同
    returnVec = [0]*len(vocabList)

    # 遍历文档中所有的单词
    for word in inputSet:

        # 如果出现了词汇表中的单词
        if word in vocabList:

            # 将输出的文档向量中的对应值设置为 1
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word %s is not in my Vocabulary!' % word)
    return returnVec

# result_1 = setOfWord2Vec(myVocabList, listOPosts[0])
# result_2 = setOfWord2Vec(myVocabList, listOPosts[3])
# print(result_1, result_2)


def trainNB0(trainMatrix, trainCatagory):
    """

    :param trainMatrix:
    :param trainCatagory:
    :return:
    """
    # 训练文档的总数
    numTrainDocs = len(trainMatrix)

    # 词汇表的长度
    numWords = len(trainMatrix[0])

    # 任意文档 属于 侮辱性 文档 的概率
    pAbusive = sum(trainCatagory)/float(numTrainDocs)

    # # 词汇表长度，以 0 填充的矩阵
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    #
    # # denom 分母项
    # p0Denom = 0.0
    # p1Denom = 0.0

    # 如果其中一个概率为0，那么最后乘积也为0
    # 为了降低这种影响，将所有词的出现数初始化为1，并将分母初始化为2
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    # 遍历训练文档集中的每一篇文档
    for i in range(numTrainDocs):
        # 如果该文档的分类为 侮辱性 文档
        if trainCatagory[i] == 1:
            # 文档矩阵相加，最后获得的 p1Num 矩阵的每个元素为该词汇在所有文档中出现的总次数
            p1Num += trainMatrix[i]
            # 矩阵单行元素相加，最后获得的 p1Denom 为整个文档集中所有词汇出现的总次数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # # 获得由每个单词 出现频率 组成的矩阵向量 p1Vect
    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom

    # 由于太多很小的数字相乘，造成 下溢出
    # 解决办法是对乘积取自然对数，通过求对数可以避免下溢出或者浮点数舍入导致的错误
    # 采用自然对数进行处理不会造成任何损失
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)

    return p0Vect, p1Vect, pAbusive

# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
# print('------------------------')
# print(trainMat)


# p0v, p1v, pAb = trainNB0(trainMat, listClasses)

# print(p0v)
# print(p0v[index_stupid])
# print('------------------------')
#
# print(p1v)
# print(p1v[index_stupid])
# print('------------------------')
#
# print(pAb)
# print('------------------------')

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """

    :param vec2Classify: 要分类的向量
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(1.0-pClass1)
    # print(p1, p0)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():

    # 训练部分
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

    # 测试部分

    # 输入的测试文档
    testEntry = ['love', 'my', 'dalmation']

    # 将 测试文档 根据 词汇表 转化为 矩阵向量
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))


# testingNB()

def bagOfWords2VecMN(vocabList, inputSet):
    # 创建一个其中所含元素都为 0 的向量，长度与词汇表相同
    returnVec = [0] * len(vocabList)

    # 遍历文档中所有的单词
    for word in inputSet:

        # 如果出现了词汇表中的单词
        if word in vocabList:

            # 将输出的文档向量中的对应值设置为 1
            returnVec[vocabList.index(word)] += 1
    return returnVec