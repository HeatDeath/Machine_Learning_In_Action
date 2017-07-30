# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from pylab import *


mpl.rcParams['font.sans-serif'] = ['SimHei']


# 定义文本框 和 箭头样式 style

# 决策节点 注释框 style 为 sawtooth（锯齿）， fc 控制注释框内的颜色深度
decisionNode = dict(boxstyle="sawtooth", fc="0.8")

# 叶子节点 注释框的 boxstyle 为 round4 ，fc控制颜色深度
leafNode = dict(boxstyle="round4", fc="0.8")

# 箭头的 arrowstyle 为 <-
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制带箭头的注解
    :param nodeTxt: 节点文本
    :param centerPt: 箭头指向点
    :param parentPt: 箭头起始点
    :param nodeType: 节点类型
    :return:
    """
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrow_args)

# def createPlot():
#     # 创建一个新图形
#     fig = plt.figure(1, facecolor='white')
#
#     # 清空绘图区
#     fig.clf()
#
#     createPlot.axl = plt.subplot(111, frameon=False)
#
#     # 绘制节点
#     plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()

# createPlot()

def getNumLeafs(myTree):
    """
    遍历整棵树，累计叶子节点的个数，并返回该值
    :param myTree: 输入的决策树
    :return: 叶子节点的总数
    """
    # 统计叶子节点数量
    numLeafs = 0

    # 第一个关键字是第一次划分数据集的类别标签
    firstStr = list(myTree.keys())[0]

    # 附带的数值表示子节点的取值
    secondDict = myTree[firstStr]

    # 遍历子节点的 key
    for key in secondDict.keys():

        # 如果子节点是字典类型，则该节点也是一个判断节点
        if type(secondDict[key]).__name__ == 'dict':

            # 递归调用 getNumLeafs()
            numLeafs += getNumLeafs(secondDict[key])

        # 否则该节点为 叶子节点
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    计算遍历过程中遇到的判断节点个数（树的深度）
    :param myTree: 输入的决策树
    :return: 树的深度
    """
    # 决策树的最大深度
    maxDepth = 0

    # 第一个关键字
    firstStr = list(myTree.keys())[0]

    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    """
    用于测试，返回预定义的树结构
    :param i:
    :return:
    """
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

myTree = retrieveTree(0)
# numLeafs = getNumLeafs(myTree)
# treeDepth = getTreeDepth(myTree)

# print(myTree, numLeafs, treeDepth)

def createPlot(inTree):
    # 创建一个新图形
    fig = plt.figure(1, facecolor='white')

    # 清空绘图区
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)

    """
     1、全局变量 plotTree.totalW 和 plotTree.totalD 分别存储树的 宽度 和 深度（高度）
      
     2、使用这两个变量计算树节点的摆放位置，这样可以将树绘制在水平方向和竖直方向的中心位置
      
     3、树的宽度 plotTree.totalW 用于计算放置判断节点的位置，主要的计算原则是将它放在所有叶子节点的中间，
    而不仅仅是它子节点的中间
    """
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))

    """
    全局变量 plotTree.xOff 和 plotTree.yOff 追踪已经绘制的节点位置，以及放置下一个节点的恰当位置
    """
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0

    """
        实际输出图形中并没有 x、y 坐标，通过计算树包含的所有叶子节点数，划分图形的宽度，从而计算得到当前节点
    的中心位置，也就是说，我们按照叶子节点的数目将 x 轴划分为若干部分。
        按照图形比例绘制树图形的好处是无需关心实际输出图形的大小，一旦图形发大小生了变化，函数会自动按照图形大
    小重新绘制。
    """
    plotTree(inTree, (0.5, 1.0), '')

    plt.show()

def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点中间，填充文本信息
    :param cntrPt: 箭头指向点坐标
    :param parentPt: 箭头起始点坐标
    :param txtString: 文本信息
    :return:
    """
    # 求得中点坐标位置
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.axl.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD



# myTree['no surfacing'][3] = 'maybe'
# createPlot(myTree)











