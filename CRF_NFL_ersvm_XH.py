# Copyright 2016-2024, Yanlin Duan, comdyling2016@163.com; Shuyin Xia,xia_shuyin@outlook;
# 生成完全随机树：用于噪声检测

from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import *
import numpy as np
from collections import Counter
# from sklearn.neighbors import KNeighborsClassifier as kNN
# from keras.models import Sequential  # 一种是CNN比较常用到的sequential网络结构
##from keras.layers.core import Dense, Dropout, Activation
##from keras.models import load_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier  # 分类
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
# 绘图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy


class BinaryTree:
    def __init__(self, labels=array([]), datas=array([])):
        self.label = labels
        self.data = datas
        self.leftChild = None
        self.rightChild = None

    def set_rightChild(self, rightObj):
        self.rightChild = rightObj

    def set_leftChild(self, leftObj):
        self.leftChild = leftObj

    def get_rightChild(self):
        return self.rightChild

    def get_leftChild(self):
        return self.leftChild

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label


# 输出中文,用于python2
def print_(hanzi):
    print((hanzi).decode('utf-8'))


# 将data 以第splitAttribute列元素的splitValue为界划分成leftData 和rightData两部分
def splitData(data, splitAttribute, splitValue):
    leftData = array([])
    rightData = array([])
    for c in data[:, ]:
        if c[splitAttribute] > splitValue:
            if len(rightData) == 0:
                rightData = c
            else:
                rightData = vstack((rightData, c))
        else:
            if len(leftData) == 0:
                leftData = c
            else:
                leftData = vstack((leftData, c))

    return leftData, rightData


# data 为二维矩阵数据
# 第一列为标签[0,1]，或者[-1,1]
# 最后一列为样本序数
# 返回一个树的根节点
minNumSample = 10


def generateTree(data, uplabels=[]):
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        numberSample = 1
        numberAttribute = data.size

    if numberAttribute == 0:
        return None

    numberAttribute = numberAttribute - 2

    # 当前数据的类别，也叫节点类别
    labelNumKey = []
    if numberSample == 1:
        labelvalue = data[0]
        rootdata = data[numberAttribute + 1]
    else:
        # labelAttribute=data[:,0]
        labelNum = Counter(data[:, 0])
        labelNumKey = list(labelNum.keys())
        labelNumValue = list(labelNum.values())
        labelvalue = labelNumKey[labelNumValue.index(max(labelNumValue))]
        rootdata = data[:, numberAttribute + 1]

    rootlabel = hstack((labelvalue, uplabels))

    CRTree = BinaryTree(rootlabel, rootdata)

    # 树停止增长的条件至少有两个：1样本个数限制；2第一列全部相等
    if numberSample < minNumSample or len(labelNumKey) < 2:
        return CRTree
    else:
        splitAttribute = 0  # 随机得到划分属性
        splitValue = 0  # 随机得到划分属性中的值
        maxCycles = 1.5 * numberAttribute  # Maximum number of cycles
        i = 0
        while True:  # 一旦出现数据异常：除了上面两种停止树增长的条件外的异常情况，即为错误数据，这里的循环将不发停止
            i += 1
            splitAttribute = random.randint(
                1, numberAttribute)  # 函数返回包括范围边界的整数
            if splitAttribute > 0 and splitAttribute < numberAttribute + 1:  # 符合矩阵要求的属性列
                dataSplit = data[:, splitAttribute]
                # uniquedata=list(Counter(dataSplit).keys()) #作用同下面一行
                uniquedata = list(set(dataSplit))
                if len(uniquedata) > 1:
                    break
            if i > maxCycles:  # 数据异常导致的树停止增长
                # print('数据异常')
                return CRTree
        sv1 = random.choice(uniquedata)
        i = 0
        while True:
            i += 1
            sv2 = random.choice(uniquedata)
            if sv2 != sv1:
                break
            if i > maxCycles:
                # print('查找划分点超时')
                return CRTree
        splitValue = mean([sv1, sv2])
        leftdata, rightdata = splitData(data, splitAttribute, splitValue)
        CRTree.set_leftChild(generateTree(leftdata, rootlabel))
        CRTree.set_rightChild(generateTree(rightdata, rootlabel))
        return CRTree


# 调用函数
def CRT(data):
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        return None
    orderAttribute = arange(numberSample).reshape(numberSample, 1)
    data = hstack((data, orderAttribute))
    completeRandomTree = generateTree(data)
    return completeRandomTree


# 返回两行N列的矩阵，第一行是样本标签，第二行是判断噪声阈值
def visitCRT(tree):
    if tree.get_leftChild() == None and tree.get_rightChild() == None:
        data = tree.get_data()
        labels = checkLabelSequence(tree.get_label())
        try:
            labels = zeros(len(data)) + labels
        except TypeError:
            pass
        result = vstack((data, labels))
        return result
    else:
        resultLeft = visitCRT(tree.get_leftChild())
        resultRight = visitCRT(tree.get_rightChild())
        result = hstack((resultLeft, resultRight))
        return result


# 返回一个序列最近两次变化之间的个数
def checkLabelSequence(labels):
    index1 = 0
    for i in range(1, len(labels)):
        if labels[index1] != labels[i]:
            index1 = i
            break
    if index1 == 0:
        return 0

    index2 = 0
    for i in range(index1 + 1, len(labels)):
        if labels[index1] != labels[i]:
            index2 = i
            break
    if index2 == 0:
        index2 = len(labels)
    return index2 - index1


# 返回是否是噪声数据的序列——树
def filterNoise(data, tree=None, niThreshold=3):
    if tree == None:
        tree = CRT(data)
    visiTree = visitCRT(tree)
    visiTree = visiTree[:, argsort(visiTree[0, :])]
    for i in range(len(visiTree[0, :])):
        if visiTree[1, i] >= niThreshold:  # 是噪声
            visiTree[1, i] = 1
        else:
            visiTree[1, i] = 0
    return visiTree[1, :]


# 返回是否是噪声数据的序列——森林
def CRFNFL(data, ntree=100, niThreshold=3):
    m, n = data.shape
    result = zeros((m, ntree))
    for i in range(ntree):
        visiTree = filterNoise(data, niThreshold=niThreshold)
        result[:, i] = visiTree

    noiseData = []
    for i in result:
        if sum(i) >= 0.5 * ntree:
            noiseData.append(1)
        else:
            noiseData.append(0)

    return array(noiseData)


# 删除异常数据
def deleteNoiseData(data, noiseOrder):
    flag = 0
    for i in range(noiseOrder.size):
        if noiseOrder[i] == 0:
            if flag == 0:
                redata = data[i, :]
                flag = 1
            else:
                redata = vstack((redata, data[i, :]))
    return redata


'''
标Func的是未去噪方法
'''


def svmFunc(traindata, testdata):
    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]

    model = svm.SVC(kernel='rbf', gamma='auto', C=256)  # linear
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel)  # 预测精度
    return precision


def svmFunc1(traindata, testdata):
    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]

    model = svm.SVC(kernel='rbf', gamma='auto')  # linear
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel)  # 预测精度
    return precision


'''
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
'''


# 结合噪声检测的分类算法：


def CRFNFL_SVM(traindata, Validationdata, testdata, _ntree, _niThreshold, _data_name='', _noise_rate=''):
    ntree = _ntree
    niThreshold = _niThreshold
    data_name = _data_name
    noise_rate = _noise_rate
    print('ntree', _ntree, '_nithreshold', _niThreshold)
    # minSubNi = ntree // 10
    # if(minSubNi<5):
    #     minSubNi = 11

    # 建立ntree棵树
    buildTreeTime1 = time.time()
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0
    try:
        m, n = traindata.shape
    except ValueError as e:
        print(str(e))
        return 0

    forest = array([])  # m行ntree列的矩阵
    for i in range(ntree):
        tree = CRT(traindata)  # 生成CRDTs
        visiTree = visitCRT(tree)  # 计算噪声强度：第一行为标签，第二行为对应的噪声强度ni
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))
    buildTreeTime2 = time.time()
    buildTreeTime = buildTreeTime2 - buildTreeTime1
    precision = 0
    timeInit = 999999
    minTimeNi = None
    minTimeVote = None
    minTimePrecision = None
    # 森林规模小于10，只讨论niThreshold的变化情况下的最优精度
    # if ntree < 10:
    if ntree < 10:
        print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        # # 用不同的niThreshold检测噪声数据，更新最优精度
        # # sub niThreshold = i
        # # 使用不同的vote,寻优
        # # for voteNum in range(1, ntree+1):
        # # voteNum = 4
        # time0 = time.time()
        # noiseForest = zeros(m)  # 保存噪声检测结果
        # # 开始遍历forest矩阵
        # for j in range(m):  # 一个数据的检测过程
        #     for k in range(ntree):
        #         if forest[j, k] >= 5:
        #             noiseForest[j] += 1
        #     if noiseForest[j] >= 4.5:
        #     # if noiseForest[j] >= voteNum:  # votes
        #         noiseForest[j] = 1
        #     else:
        #         noiseForest[j] = 0
        time0 = time.time()
        # for subNi in range(2, niThreshold + 1):
        # sub niThreshold = i
        # 用不同的niThreshold检测噪声数据，更新最优精度
        # noiseForest = zeros(m)  # 保存噪声检测结果
        # 开始遍历forest矩阵
        # for j in range(m):  # 一个数据的检测过程
        #     for k in range(ntree):
        #         if forest[j, k] >= 5:
        #             noiseForest[j] += 1
        #
        #     if noiseForest[j] >= 0.5 * ntree:  # votes
        #         noiseForest[j] = 1
        #     else:
        #         noiseForest[j] = 0
        for subNi in range(2, niThreshold + 1):  # 设置噪声强度阈值,subNi=NI
            for voteNum in range(1, ntree + 1):  # 设置投票结果的数量,VR=voteNum
                # time0 = time.time()
                noiseForest = zeros(m)  # 保存噪声检测结果
                # 开始遍历forest矩阵
                for j in range(m):  # 一个数据的检测过程
                    for k in range(ntree):
                        if forest[j, k] >= 5:
                            noiseForest[j] += 1

                    if noiseForest[j] >= voteNum:  # votes
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                    try:
                        denoiseTraindata = deleteNoiseData(
                            traindata, noiseForest)
                    except:
                        continue
                time1 = time.time()
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # # 验证精度
                preTemp = svmFunc(denoiseTraindata,
                                Validationdata)  # 调用系统基本分类算法
                time2 = time.time()
                delNoiseTime = time1 - time0
                delNoiseSvmTime = time2 - time1
                # print(c)
                if timeInit > delNoiseSvmTime:
                    timeInit = delNoiseSvmTime
                    minTimeNi = 5
                    # minTimeVote = voteNum
                    minTimePrecision = precision
                if precision < preTemp:
                    precision = preTemp
                # '投票阈值', voteNum
                print('噪声阈值', voteNum, 'Ni值', subNi, '精度', preTemp, '建树时间',
                    buildTreeTime, '检测并删除噪声时间', delNoiseTime, '删除噪声后SVM时间', delNoiseSvmTime,
                    '总时间', buildTreeTime + delNoiseTime + delNoiseSvmTime, '数据名', data_name)

                # 追加数据
                # r_xls = open_workbook('testDataNI5.xls')  # 读取excel文件
                # row = r_xls.sheets()[0].nrows  # 获取已有的行数
                # excel = copy(r_xls)  # 将xlrd的对象转化为xlwt的对象
                # table = excel.get_sheet(0)  # 获取要操作的sheet
                # table.write(row, 0, 5)  # 括号内分别为行数、列数、内容
                # # table.write(row, 1, voteNum)  # 括号内分别为行数、列数、内容
                # table.write(row, 2, preTemp)  # 括号内分别为行数、列数、内容
                # table.write(row, 3, buildTreeTime)  # 括号内分别为行数、列数、内容
                # table.write(row, 4, delNoiseTime)  # 括号内分别为行数、列数、内容
                # table.write(row, 5, delNoiseSvmTime)  # 括号内分别为行数、列数、内容
                # table.write(row, 6, buildTreeTime+delNoiseTime+delNoiseSvmTime)  # 括号内分别为行数、列数、内容
                # table.write(row, 7, data_name)  # 括号内分别为行数、列数、内容
                # # excel.save('testDataNI5_vot4.xls')  # 保存并覆盖文件

        tm = buildTreeTime + delNoiseTime + delNoiseSvmTime  # 总时间
        print('总时间', tm)
        tm1 = delNoiseSvmTime  # 分类时间
        print('分类时间', tm1)
        tm2 = buildTreeTime + delNoiseTime  # 去噪时间
        print('去噪时间', tm2)

    else:  # 森林规模大于10，每10棵树作为一个间隔遍历不同规模下的精度
        startNtree = 1
        endNtree = ntree // 10
        remainderNtree = ntree % 10  # 如果有余数，最后在用所有树更新一次精度
        # 每10棵树作为一个间隔遍历不同规模下的精度
        d = []
        for i in range(startNtree, endNtree + 1):
            print('i', i)
            subNtree = i * 10
            # print('subNtree:',subNtree)
            # 将subForest 作为森林进行不同niThreshold的遍历
            # for subNi in range(2, niThreshold + 1, 2):
            for subNi in range(2, niThreshold + 1, 2):
                print('subNi', subNi)
                # 使用不同的vote,寻优
                for voteNum in range(1, 10):
                    print('vote', voteNum)
                    noiseForest = zeros(m)
                    # 检测噪声
                    try:
                        for j in range(m):
                            for k in range(subNtree):
                                # if forest[j, k] >= subNi:
                                if forest[j, k] >= subNi:
                                    noiseForest[j] += 1

                            # 判定噪声
                            if noiseForest[j] >= voteNum:
                                noiseForest[j] = 1
                            else:
                                noiseForest[j] = 0
                            # print(noiseForest)
                        # print('噪声', noiseForest)
                        # denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                        denoiseTraindata = deleteNoiseData(
                            traindata, noiseForest)
                        # 验证精度
                        time1 = time.time()
                        # # 验证精度
                        preTemp = svmFunc(denoiseTraindata,
                                          Validationdata)  # 调用系统基本分类算法
                        time2 = time.time()
                        c = time2 - time1
                        # if precision < preTemp:
                        #     best_ntree = subNtree
                        #     best_ni = subNi
                        #     best_vote = voteNum
                        # minTimePrecision = precision
                        if precision < preTemp:
                            precision = preTemp
                            best_ntree = subNtree
                            best_ni = subNi
                            best_vote = voteNum
                        print('subNi', subNi, 'voteNum', voteNum, 'preTemp', preTemp, 'c', c, 'max_precision',
                              precision)
                        # 追加数据
                        r_xls = open_workbook('CRF_SVM_寻优2.xls')  # 读取excel文件
                        row = r_xls.sheets()[0].nrows  # 获取已有的行数
                        excel = copy(r_xls)  # 将xlrd的对象转化为xlwt的对象
                        table = excel.get_sheet(0)  # 获取要操作的sheet
                        table.write(row, 0, data_name)  # 括号内分别为行数、列数、内容
                        # table.write(row, 1, voteNum)  # 括号内分别为行数、列数、内容
                        table.write(row, 1, noise_rate)  # 括号内分别为行数、列数、内容
                        table.write(row, 2, preTemp)  # 括号内分别为行数、列数、内容
                        table.write(row, 3, subNtree)  # 括号内分别为行数、列数、内容
                        table.write(row, 4, subNi)  # 括号内分别为行数、列数、内容
                        table.write(row, 5, voteNum)  # 括号内分别为行数、列数、内容
                        # table.write(row, 7, data_name)  # 括号内分别为行数、列数、内容
                        excel.save('CRF_SVM_寻优2.xls')  # 保存并覆盖文件
                        # d.append([data_name, noise_rate, preTemp, subNtree, subNi, voteNum])
                    except ValueError:
                        continue
                    except UnboundLocalError:
                        continue
                    # return precision
        if remainderNtree > 0:
            for subNi in range(2, niThreshold + 1):
                for voteNum in range(2, remainderNtree + 1):
                    print('remainderNtree', voteNum)
                    noiseForest = zeros(m)
                    for j in range(m):
                        for k in range(ntree):
                            # if forest[j, k] >= subNi:
                            if forest[j, k] >= 5:
                                noiseForest[j] += 1

                        if noiseForest[j] >= voteNum:
                            noiseForest[j] = 1
                        else:
                            noiseForest[j] = 0

                    denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                    # # 验证精度
                    time1 = time.time()
                    # print('denoiseTraindata.shape',denoiseTraindata.shape)
                    # # 验证精度
                    preTemp = svmFunc(denoiseTraindata,
                                      Validationdata)  # 调用系统基本分类算法
                    time2 = time.time()
                    c = time2 - time1

                    if precision < preTemp:
                        precision = preTemp
                    if timeInit > c:
                        timeInit = c
                    print('subNi', subNi, 'voteNum', voteNum, 'preTemp',
                          preTemp, 'c', c, 'minTime', timeInit)
                    # return precision
    # print('minTimePrecision', minTimePrecision, 'minTimeNi', minTimeNi, 'minTimeVote', minTimeVote, 'minTime', timeInit)
    print('最优Ntree:', best_ntree, '最优Ni:', best_ni,
          '最优vote:', best_vote, '最优精度:', precision)
    return precision, best_ntree, best_ni, best_vote


'''''
接口
'''''
def crfnfl_all(traindata, testdata, Validationdata, _ntree, _niMax, _data_name, _noise_rate):
    ntree = _ntree
    niMax = _niMax
    data_name = _data_name
    # d1 =d
    print('niMax', niMax)
    m, n = traindata.shape
    print('traindata:', m, n)
    m, n = testdata.shape
    print('testdata:', m, n)
    print("原始精度")
    svmTime1 = time.time()
    pre3 = svmFunc1(traindata, testdata)
    svmTime2 = time.time()
    svmTime = svmTime2 - svmTime1
    print("svm = :", pre3, '未删除噪声svm时间', svmTime)


    pre8, Ntree, Ni, Vote = CRFNFL_SVM(traindata, Validationdata, testdata, _ntree=ntree, _niThreshold=niMax, _data_name=data_name,
                                    _noise_rate=_noise_rate)
    print("去噪后精度")
    print(pre8)
    return pre3, pre8, Ntree, Ni, Vote



if __name__ == '__main__':
    # 操作数据集，划分训练集，测试集，验证集
    data_name = ['sonar']
    # , 'breastcancer', 'creditApproval', 'fourclass', 'svmguide1', 'svmguide3']
    # ['sonar', 'votes', 'clean1', 'breastcancer', 'creditApproval', 'diabetes', 'fourclass',
    #          'splice', 'svmguide3', 'isolet5', 'madelon', 'isolet1234', 'svmguide1', 'mushrooms']
    noise_rate = ['0.2']
    # ['0.05', '0.1', '0.15', '0.2']

    # , 'creditApproval', 'splice', 'svmguide3',
    #              'madelon', 'svmguide1' , 'mushrooms', 'codrna','ijcnn1']
    # 读取文件路径
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('My Worksheet')

    # 写入excel
    # 参数对应 行, 列, 值
    worksheet.write(0, 0, label='数据名')
    worksheet.write(0, 1, label='噪声率')
    worksheet.write(0, 2, label='精度')
    worksheet.write(0, 3, label='Ntree')
    worksheet.write(0, 4, label='Ni')
    worksheet.write(0, 5, label='Voting')
    # 保存
    workbook.save('CRF_SVM_寻优2.xls')
    a = []
    for dn in data_name:
        for nr in noise_rate:
            # f = open(r'/Users/xiah/PycharmProjects/practice_1/svm_test/Ni5CRF/datasets_0.2noise/%s_0.2.csv' % data_name[0])
            df = pd.read_csv(r'datasets/datasets_0.2noise/%s_%s.csv' %
                            (dn, nr), header=0)
            # df = pd.read_csv(r'/Users/xiaoliang/Desktop/othercode/bgw/dataset_noise/{} {}%.csv'.format(data_name[d], 20), header=None)
            # df = pd.read_csv(f, header=None)
            print(dn, nr)
            Row = df.shape[0]  # 获取数据行数   v
            Column = df.shape[1]  # 特征数，可以确定ni最大值
            print('Row, Column', Row, Column)
            dfNum = int(0.8 * Row)
            trainNum = int(12 / 15 * dfNum)
            df1 = df.iloc[:dfNum, :]
            train = df1.iloc[:trainNum, :].values
            Validationdata = df1.iloc[trainNum:, :].values
            test = df.iloc[dfNum:, :].values
            # 记录数据，创建一个workbook 设置编码
            # workbook = xlwt.Workbook(encoding='utf-8')
            # # 创建一个worksheet
            # worksheet = workbook.add_sheet('My Worksheet')
            #
            # 0.7692307692307693
            # # 写入excel
            # # 参数对应 行, 列, 值
            # worksheet.write(0, 0, label='噪声阈值')
            # worksheet.write(0, 1, label='投票阈值')
            # worksheet.write(0, 2, label='精度')
            # worksheet.write(0, 3, label='建树时间')
            # worksheet.write(0, 4, label='检测并删除噪声时间')
            # worksheet.write(0, 5, label='删除噪声后SVM时间')
            # worksheet.write(0, 6, label='总时间')
            # worksheet.write(0, 7, label='数据名')
            # # 保存
            # workbook.save('testDataNI5.xls')
            returnSvm, returnCRFNFL_SVM, Ntree, Ni, Vote = crfnfl_all(train, test, Validationdata, _ntree=100, _niMax=int(Column / 2) + 1,
                                                                    _data_name=dn, _noise_rate=nr)
            a.append([dn, nr, returnSvm, returnCRFNFL_SVM, Ntree, Ni, Vote])
    column = ['datasets', '噪声率', '原始精度',
            '去噪后精度', '最优Ntree', '最优Ni', '最优Voting']
    d1 = pd.DataFrame(a, columns=column)
    print(d1)
    d1.to_excel(r'results/CRF_erSVM_寻优2.xls', index=False)
    # 遍历数据集
    # for item in data_name:
    #     returnSvm, returnCRFNFL_SVM = crfnfl_all(train, test, Validationdata
    #                                              , _ntree=9, _niMax=int(Column/2)+1, _data_name=item)a.a
    #  0.7396428823487896
    #  0.7431520444620882
