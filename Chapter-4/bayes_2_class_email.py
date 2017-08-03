from bayes import createVocabList, setOfWord2Vec, trainNB0, classifyNB
import random
import numpy as np

def textParse(bigString):
    """
    接受一个 bigString 长字符串，并将其解析为 由长度大于 2 单词 组成的 list
    :param bigString: 长字符串
    :return: 单词组成的 list
    """
    import re

    # 以 [a-zA-Z0-9] 以外的元素进行 拆分
    listOfTokens = re.split('\W+', bigString)

    # 将长度大于 2 的单词转换为小写，并存入 list
    return [tok.lower for tok in listOfTokens if len(tok) > 2]

def spamTest():
    """
    对贝叶斯辣鸡邮件分类器进行自动化处理
    :return:
    """
    # 初始化 文档 list， list 中的每一个元素都是一个 文档（由单词组成的 list）
    docList = []

    # 初始化 文档分类 list， classList 与 docList 中的每个元素 一一对应，即为对应 文档的分类
    classList = []

    # 初始化 全部文本 list， list 中的每个元素， 为 一个单词
    fullText = []

    # 遍历 spam 和 ham 目录下的各个 txt 文件
    for i in range(1, 26):
        # 打开目录下的一个 文本 ，并对其 进行解析 为 文档
        wordList = textParse(open('email/spam/%d.txt' % i).read())

        # 将文档 append 入 docList 中
        docList.append(wordList)

        # 将文档 extend 到 fullText 后
        fullText.extend(wordList)

        # 在 classList 中 添加 文档对应的分类
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 根据 docList 调用 createVocabList 创建 词汇表
    vocabList = createVocabList(docList)

    # 初始化 trainingSet 训练集，一个长度为 50 的 list
    trainingSet = list(range(50))

    # 初始化 testSet 测试集，为空
    testSet = []

    # 重复 10 次
    for i in range(10):
        # 从 0 到 训练集长度，随机选择一个整数，作为 randIndex 随机索引
        randIndex = int(random.uniform(0, len(trainingSet)))

        # 测试集 添加 训练集中随机索引 对应的 元素
        testSet.append(trainingSet[randIndex])

        # 从 训练集 中 删除 随机索引 对应的元素
        del(trainingSet[randIndex])

    # 初始化 训练矩阵
    trainMat = []

    # 初始化 训练分类 list
    trainClasses = []

    # 依次遍历 训练集 中的每个元素， 作为 docIndex 文档索引
    for docIndex in trainingSet:
        # 在 trainMat 训练矩阵中 添加 单词向量 矩阵
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        # 在 trainClasses 训练文档分类中 添加 文档对应的分类
        trainClasses.append(classList[docIndex])

    # 调用 trainNB0 函数，以 trainMat 和 trainClasses 作为输入数据，计算 p0V, p1V, pSpam
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    # 初始化 错误统计
    errorCount = 0

    # 遍历 测试集 中的每个元素 作为 文档索引 docIndex
    for docIndex in testSet:
        # 生成单词向量
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])

        # 如果计算后的分类结果 与 实际分类 不同
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 错误数量 + 1
            errorCount += 1

    # 打印 错误率
    print('the error rate is:', float(errorCount)/len(testSet))

'''
随机选择数据的一部分作为训练集，而剩余部分作为测试集的过程称为 留存交叉验证
'''

# spamTest()

