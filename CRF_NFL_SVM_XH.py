# Copyright 2016-2024, Yanlin Duan, comdyling2016@163.com; Shuyin Xia,xia_shuyin@outlook;
# 生成完全随机树：用于噪声检测

import numpy
from numpy import *
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier as kNN
##from keras.models import Sequential  # 一种是CNN比较常用到的sequential网络结构
##from keras.layers.core import Dense, Dropout, Activation
##from keras.models import load_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier  # 分类
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
import pandas as pd
import time



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
            splitAttribute = random.randint(1, numberAttribute)  # 函数返回包括范围边界的整数
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

    model = svm.SVC(kernel='rbf', gamma='auto')  # linear
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel)  # 预测精度
    return precision

def CRFNFL_SVM(traindata, Validationdata, testdata, ntree, niThreshold, _data_name=''):
    print(ntree, niThreshold)
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
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))
    buildTreeTime2 = time.time()
    buildTreeTime = buildTreeTime2 - buildTreeTime1
    precision = 0
    time0 = 9999
    if ntree < 10:  # 森林规模小于10，只讨论niThreshold的变化情况下的最优精度
        print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        time0 = time.time()
        # for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # 用不同的niThreshold检测噪声数据，更新最优精度
        noiseForest = zeros(m)  # 保存噪声检测结果
        # 开始遍历forest矩阵
        for j in range(m):  # 一个数据的检测过程
            for k in range(ntree):
                if forest[j, k] >= 5:
                    noiseForest[j] += 1

            if noiseForest[j] >= 0.5 * ntree:  # votes
                noiseForest[j] = 1
            else:
                noiseForest[j] = 0

        denoiseTraindata = deleteNoiseData(traindata, noiseForest)
        # print('denoiseTraindata.shape',denoiseTraindata.shape)
        # # 验证精度
        time1 = time.time()
        preTemp = svmFunc(denoiseTraindata, Validationdata)  # 调用系统基本分类算法
        # 测试精度
        time2 = time.time()
        delNoiseTime = time1 - time0
        delNoiseSvmTime = time2 - time1#去噪后分类时间
        # tm1 = time2 - time1
        # print(tm)
        # preTest = svmFunc(denoiseTraindata, testdata)
        # print("preTest:",preTest)
        # print('preTemp',preTemp)
        if precision < preTemp:
            precision = preTemp
        if time0 > delNoiseSvmTime:
            time0 = delNoiseSvmTime
        print('删除噪声时间', delNoiseTime)
        totaltime = buildTreeTime + delNoiseTime + delNoiseSvmTime
    else:  # 森林规模大于10，每10棵树作为一个间隔遍历不同规模下的精度
        startNtree = 1
        endNtree = ntree // 10
        remainderNtree = ntree % 10  # 如果有余数，最后在用所有树更新一次精度
        # 每10棵树作为一个间隔遍历不同规模下的精度
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # print('subNtree:',subNtree)
            # 将subForest 作为森林进行不同niThreshold的遍历
            for subNi in range(2, niThreshold + 1, 2):
                noiseForest = zeros(m)
                try:
                    for j in range(m):
                        for k in range(subNtree):
                            if forest[j, k] >= subNi:
                                noiseForest[j] += 1

                        if noiseForest[j] >= 0.5 * subNtree:
                            noiseForest[j] = 1
                        else:
                            noiseForest[j] = 0

                    denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                    # print('denoiseTraindata.shape',denoiseTraindata.shape)
                    # 验证精度
                    time3 = time.time()
                    preTemp = svmFunc(denoiseTraindata, Validationdata)  # 基本分类器分类结果
                    time4 = time.time()
                    tm1 = time4 - time3
                    print(preTemp, tm1)
                    # print(preTemp)
                    # 测试精度
                    # preTest = svmFunc(denoiseTraindata, testdata)
                    # print("preTest:",preTest)
                    # print('preTemp',preTemp)
                    if precision < preTemp:
                        precision = preTemp
                        best_ntree = subNtree
                        best_ni = subNi
                    if time0 > tm1:
                        time0 = tm1
                    print('subNi', subNi, 'preTemp', preTemp, 'c', tm1, 'max_precision', precision)
                except ValueError:
                    continue
                except UnboundLocalError:
                    continue
        if remainderNtree > 0:
            # print('remainderNtree:',remainderNtree)
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * ntree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # # 验证精度
                preTemp = svmFunc(denoiseTraindata, Validationdata)
                # 测试精度
                # preTest = svmFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp
    print('最优Ntree:', best_ntree, '最优Ni:', best_ni, '最优精度:', precision)

    return precision, best_ntree, best_ni

'''''
接口
'''''
def crfnfl_all(traindata, testdata, Validationdata, ntree, ni):
    m, n = traindata.shape
    print('traindata:', m, n)
    m, n = testdata.shape
    print('testdata:', m, n)
    print("原始精度")
    pre3 = svmFunc(traindata, testdata)

    print("svm = :", pre3)
    # a = []


    print("去噪后精度")
    time_1 = time.time()
    pre8, Ntree, Ni = CRFNFL_SVM(traindata,  Validationdata, testdata, ntree, ni)
    time_2 = time.time()

    # tm_1 = time_2 - time_1
    print("svm = :", pre8)

    # a.append([pre3,pre8])
    # print(a)
    return pre3, pre8, Ntree, Ni


if __name__ == '__main__':
    import numpy as np

    data_name = ['sonar', 'votes','clean1', 'breastcancer','creditApproval','diabetes', 'fourclass',
                 'splice','svmguide3','isolet5','madelon','isolet1234','svmguide1','mushrooms']
        # , 'mushrooms', 'codrna', 'ijcnn1']
    # noise_rate = ['0.05', '0.1', '0.15', '0.2']
    noise_rate = [ '0.2']
    a = []
    for dn in data_name:
        for nr in noise_rate:
            # df = pd.read_csv('E:/123/crf-nfl-3.0-src/datasets/datasets_0.2noise/%s_%s.csv' %(dn, nr), header=0)
            df = pd.read_csv(r'datasets/datasets_0.2noise/%s_%s.csv' %(dn, nr), header=0)
            
            # data = df.values
            Row = df.shape[0]  # 获取数据样本数
            Column = df.shape[1]  # 特征数，可以确定ni最大值
            print(dn,nr)
            print('Row, Column', Row, Column)

            dfNum = int(0.8 * Row)
            trainNum = int(12 / 15 * dfNum)
            df1 = df.iloc[:dfNum, :]
            train = df1.iloc[:trainNum, :].values
            Validationdata = df1.iloc[trainNum:, :].values
            test = df.iloc[dfNum:, :].values
            # traindata_1, testdata_1 = np.split(data, [int(.8 * len(data))])
            # traindata_2, Validationdata_2 = np.split(traindata_1, [int(.8 * len(traindata_1))])
            _ntree = 100
            _niMax = int(Column / 2) + 1
            pr1, pr2, Ntree, Ni = crfnfl_all(train, test, Validationdata, _ntree, _niMax)
            # tm = tm_1 - tm_2
            a.append([dn,nr, pr1, pr2, Ntree, Ni])
    columns = ['datasets', '噪声率', '原始精度', '去噪后精度', '最优Ntree', '最优Ni']
    d1 = pd.DataFrame(a, columns=columns)
    print(d1)
    d1.to_excel(r'results/CRF_NFL_SVM_.xls', index=False)