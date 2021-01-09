# encoding=utf-8

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score


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
    precision = model.score(testdata, testdatalabel)  # 测试精度
    print('训练精度为：',model.score(traindata, traindatalabel))
    return precision

def cross_val_scores(data):
    '''
        返回交叉验证后的平均精度
    '''
    features = data[::, 1::]
    labels = data[::, 0]
    clf = svm.SVC(kernel='rbf', gamma='auto')
    scores = cross_val_score(clf, features, labels, cv=5)
    ave_scores = np.average(scores)
    return ave_scores


if __name__ == '__main__':
    # data_name = ['breastcancer', 'codrna', 'creditApproval', 'diabetes', 'fourclass', 'isolet5',
                 #                   'ijcnn1', 'madelon', 'mushrooms', 'sonar', 'splice', 'svmguide1','clean1',
                 #                   'svmguide3', 'votes','isolet1234',] #16个数据集，未按照数据集大小排列
    # data_name = ['sonar', 'votes','clean1', 'breastcancer','creditApproval','diabetes', 'fourclass',
    #              'splice','svmguide3','isolet5','madelon','isolet1234','svmguide1','mushrooms',
    #              'codrna']
    data_name = ['sonar', 'votes','clean1', 'breastcancer','creditApproval','diabetes', 'fourclass',
                     'splice','svmguide3','isolet5','madelon','isolet1234','svmguide1']
    # data_name = ['fourclass']
    # noise_rate = [0.2]
    noise_rate = [0.05,0.1,0.15,0.20,0.25,0.3,0.35]
    a = []
    for dn in data_name:
        for nr in noise_rate:
            print(dn, nr)
            # df = pd.read_csv('E:/123/crf-nfl-3.0-src/datasets/%s_%s.csv' % (dn, nr), header=0)# 读取csv数据，并将第一行视为表头，返回DataFrame类型
            df = pd.read_csv(r'datasets/%s_%s.csv' % (dn, nr), header=0)                #TODO  可能会修改路径
        # scaler = preprocessing.MinMaxScaler()  # MinMaxScaler将样本特征值线性缩放到0.1之间
        # scaler.fit(raw_data)
        # data = scaler.transform(raw_data)
        # print(data)
            print(dn)
            raw_data = df.values    #ndarray
            print(raw_data)
            traindata, testdata = np.split(raw_data, [int(.8 * len(raw_data))])
            time_0 = time.time()
            pr1 = svmFunc(traindata, testdata)  # 分类精度
            print('分类精度：', pr1)
            time_1 = time.time()
            tm1 = time_1 - time_0  # 分类时间
            print('分类时间：', tm1)

            time_2 = time.time()
            avg_score = cross_val_scores(raw_data)  # 交叉验证平均精度
            print('交叉验证平均精度：', avg_score)
            time_3 = time.time()
            tm2 = time_3 - time_2  # 交叉验证时间
            print('交叉验证时间：', tm2)
        # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=0)
        # # 选取20%数据作为测试集，剩余为训练集
        # time_2=time.time()
        # print('Start training...')
        # clf = svm.SVC(kernel='rbf', gamma='auto')  # svm class
        # clf.fit(train_features, train_labels)# training the svc model
        # #scores = cross_val_score(clf, features, labels, cv=5)
        # time_3 = time.time()
        # print('training cost %f seconds' % (time_3 - time_2))
        #
        # print('Start predicting...')
        # test_predict=clf.predict(test_features)
        # time_4 = time.time()
        # print('predicting cost %f seconds' % (time_4 - time_3))
        # tm = time_3 - time_2
        # score = accuracy_score(test_labels, test_predict)
        # print("The accruacy score is %f" % score)
            a.append([dn, nr, pr1, avg_score, tm1, tm2])
    column = ['datasets', 'noise_rate','测试精度', '验证精度', '分类时间', '交叉验证时间']
    d1 = pd.DataFrame(a, columns=column)
    print(d1)
    # d1.to_excel('E:/123/crf-nfl-3.0-src/datasets/SVM_不同噪声.xls', index=False)
    d1.to_excel(r'results/原始SVM.csv', index=False)       #TODO


# k折交叉验证
# scores= cross_val_score(clf,features,labels,cv=5)
# time_5 = time.time()
# print(scores.mean())
# import matplotlib.pyplot as plt #可视化模块  
# #建立测试参数集  
# k_range = range(1, 31)  
# k_scores = []  
# #藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率  
# for k in k_range:  
#     knn = svm.svc()  
#     scores = cross_val_score(clf, features, labels, cv=5)  
#     k_scores.append(scores.mean())  
# #可视化数据  
# plt.plot(k_range, k_scores)  
# plt.xlabel('Value of K for KNN')  
# plt.ylabel('Cross-Validated Accuracy')  
# plt.show()  
