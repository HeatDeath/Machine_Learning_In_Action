from bayes_2_class_email import textParse
from bayes import createVocabList, trainNB0, classifyNB, bagOfWords2VecMN
import random
import feedparser
import numpy as np

ny = feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('https://sfbay.craigslist.org/stp/index.rss')



# print(ny['entries'])
# print(len(ny['entries']))

def calcMostFreq(vocabList, fullText):
    """
    遍历词汇表中的每个单词，并统计它在文本中出现的次数，根据出现次数从高到低进行排序，最后返回排序最高的 30 个单词
    :param vocabList:
    :param fullText:
    :return:
    """
    import operator

    # 高频词典 freqDict
    freqDict = {}

    # 遍历词汇表
    for token in vocabList:

        # 统计词汇表中单词的出现次数，并放入 freqDict 中
        freqDict[token] = fullText.count(token)

    # 根据字典的值进行排序（从大到小）
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)

    # 返回前 30 个单词
    return sortedFreq[:30]


def localWords(feed1, feed0):
    """

    :param feed1: newyork RSS 源
    :param feed0: sfbay RSS 源
    :return:
    """
    docList = []
    classList = []
    fullText = []

    print(len(feed0['entries']), len(feed1['entries']))
    minLen = min(len(feed0['entries']), len(feed1['entries']))

    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)

    print(len(vocabList))

    # 选择 前 30 个 高频词汇
    top30words = calcMostFreq(vocabList, fullText)

    # 遍历 高频词 list ，从 词汇表 vocabList 中 删除这 30 个高频词
    for pairW in top30words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])


    trainingSet = list(range(2*minLen))

    testSet = []

    # 从 训练集 中 随机选出 20 个 “文档” 作为 测试集
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    errorCount = 0

    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('kkk the error rate is:', float(errorCount)/len(testSet))

    return vocabList, p0V, p1V

# localWords(ny, sf)

def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")

    for item in sortedSF:
        print(str(item[0]))

    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")

    for item in sortedNY:
        print(str(item[0]))

getTopWords(ny, sf)

