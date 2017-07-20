import numpy as np
from os import listdir
from kNN import classify0

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect

# print(img2vector('testDigits/0_13.txt')[0, 32:63])
# print(listdir('testDigits'))

def handwritingClassTest():
    # 标签向量
    hwLabels = []

    # trainingDigits 目录下的文件 list
    traingingFileList = listdir('trainingDigits')

    # trainingDigits 目录下的文件个数
    m = len(traingingFileList)

    # 1*1024 由 0 填充的矩阵
    traingingMat = np.zeros((m,1024))

    # 遍历 trainingDigits 下的所有文件
    for i in range(m):
        # 获取当前文件的文件名
        fileNameStr = traingingFileList[i]

        # 获取当前文本所代表的数值
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        # 在标签向量 list 中 添加此数值
        hwLabels.append(classNumStr)

        # 训练矩阵第 i 行填充当前打开文件的 1024 个字符
        traingingMat[i,:] = img2vector('trainingDigits/{}'.format(fileNameStr))

    # testDigits 目录下的文件名称 list
    testFileList = listdir('testDigits')

    # 错误率
    errorCount = 0.0

    # 测试文件的数量
    mTest = len(testFileList)

    # 遍历测试文件
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, traingingMat, hwLabels, 3)
        print('the classifier came back with: %d,'
              'the real answer is: %d' % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print('the total number of errors is: %d' % errorCount)
    print('the total error rate is: %f' % (errorCount/float(mTest)))

# handwritingClassTest()


# the total number of errors is: 10
# the total error rate is: 0.010571
# 错误率 1.06%

def my_handwritingClassTest():
    hwLabels = []
    traingingFileList = listdir('trainingDigits')
    m = len(traingingFileList)
    traingingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = traingingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        traingingMat[i,:] = img2vector('trainingDigits/{}'.format(fileNameStr))
    testFileList = listdir('test_data')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('test_data/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, traingingMat, hwLabels, 3)
        print('the classifier came back with: %d,'
              'the real answer is: %d' % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print('the total number of errors is: %d' % errorCount)
    print('the total error rate is: %f' % (errorCount/float(mTest)))

my_handwritingClassTest()

"""
the classifier came back with: 3,the real answer is: 3
the classifier came back with: 6,the real answer is: 6
the classifier came back with: 7,the real answer is: 7
the classifier came back with: 8,the real answer is: 8
the classifier came back with: 1,the real answer is: 9
the total number of errors is: 1
the total error rate is: 0.200000

可能是因为 9 写的太细长了，以至于长得像 1？
"""