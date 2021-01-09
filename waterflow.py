# Author: Zhou Hao
# Date: 2021-01-04 11:05:01
# LastEditors: Zhou Hao
# LastEditTime: 2021-01-07 17:24:27
# Description: file content
# E-mail: 2294776770@qq.com


import pandas as pd
import numpy as np
import matplotlib.pyplot as pltsad
from imblearn import over_sampling
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_score,RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
from sklearn.metrics import confusion_matrix
import math
import datetime,os,time
import CRF_NFL_SVM_XH, CRF_NFL_ersvm_XH,SVM_XH
import time
import random
from random import choice



def read_data(path:str,choose:int,window=5,step=1):
    '''
        description:   预处理水文数据
        param {
            choose
                    : 1, 每年的11个月来预测当年9月的水流量
                    : 2,前面winodw年的9月来预测后面一年的9月流量,step=5
                    : 3,前面winodw年的9月来预测后面一年的9月流量,step自定义
            window  : choose==2时的窗口大小,默认5
            step    : 滑动的步伐,默认为1
            path    :   文件路径    #TODO
        }
        return : 预处理之后的特征X和标签Y  ndarray,int64
    '''    

    data = pd.read_csv(filepath_or_buffer=path)        #读出来的原始数据
    # print(data.head(),type(data),data.info(),data.shape)
    data_1 = data.iloc[1:,1:]       #去掉年份和不需要的信息
    # print(data_1.head(),'\n',data_1.index,data_1.columns)


    if choose == 1 :
        train_data =  data_1[data_1['MTH'] != '9']      #（671，2）
        test_data =  data_1[data_1['MTH'] == '9']       #(61,2)
        # print(train_data.head(),'\n',test_data.head())

        train_data_x = train_data.iloc[:,1]
        X = pd.DataFrame(train_data_x.values.reshape(61,11))          #先转为numpy，再用numpy的reshape
        y = pd.DataFrame(test_data.iloc[:,1].values.reshape(61,1))      
        y.columns = ['A']       #设置列名
        y['B']=y.A.apply(lambda x: 1 if int(x) >= 20000 else 0) 
        # print(y['B'])
        Y = pd.DataFrame(y['B'])

        X,Y = X.values,Y.values.T[0]     #ndarray格式
        X = X.astype(Y.dtype)     
        return X,Y
    

    elif choose == 2:
        data_9 = data_1[data_1['MTH'] == '9'].iloc[:,1] 
        # print(data_9)
        data_9 = pd.DataFrame(data_9[:60].values.reshape(int(60/(window+1)),window+1))#dataframe截取是包前不包后
        data_9.columns = ['a','b','c','d','e','f']
        data_9['g'] = data_9.e.apply(lambda x: 1 if int(x) >=20000 else 0)
        # print(data_9)

        X = data_9.iloc[:,:-2]
        Y = data_9['g']

        X,Y = X.values,Y.values     #ndarray格式
        X = X.astype(Y.dtype)  

        return X, Y
        
        
    elif choose == 3:
        data_9 = (data_1[data_1['MTH'] == '9'].iloc[:,1]).values       #ndarray，61
        # print(type(data_9),len(data_9),'\n',data_9)

        X = np.zeros(window,dtype='uint')        #创建为0的空数组
        Y = np.zeros(1,dtype='uint')
        for index in range(0,len(data_9)-window,step):
            train = data_9[index:index+window]      #包前不包后
            # print(train,'\t',index,'---',index+window,'\t',data_9[index],
            #         data_9[index+window-1],'\t',data_9[index+window]) 
            X = np.vstack((X,train))        #纵轴拼接数组
            
            test = data_9[index+window]
            Y = np.hstack((Y,test))

        X,Y = X[1:].astype('int64'), Y[1:].astype('int64')
        # print(Y)
        Y = np.where(Y>=20000,1,0)
        # print(Y)


        return X,Y               


def svm(raw_data,data_set,noise_rate):
    '''
        raw_data:ndarray,int64
    '''

    traindata, testdata = np.split(raw_data, [int(.8 * len(raw_data))])     #切割数据集
    time_0 = time.time()
    pr1 =SVM_XH.svmFunc(traindata, testdata)  # 分类精度
    # print('分类精度：', pr1)
    time_1 = time.time()
    tm1 = time_1 - time_0  # 分类时间
    # print('分类时间：', tm1)
    

    time_2 = time.time()
    avg_score =SVM_XH.cross_val_scores(raw_data)  # 交叉验证平均精度
    print('交叉验证平均精度：', avg_score)
    time_3 = time.time()
    tm2 = time_3 - time_2       # 交叉验证时间
    print('交叉验证时间：', tm2)
    

    '''保存每次运行的结果'''
    # results = []
    # results.append([pr1, avg_score, tm1, tm2])
    # column = ['测试精度', '验证精度', '分类时间', '交叉验证时间']
    # d1 = pd.DataFrame(results, columns=column)
    # print(d1)
    
    # curtime = datetime.datetime.now()
    # time_str = datetime.datetime.strftime(curtime,'%Y-%m-%d=%H-%M-%S')
    # d1.to_csv(r'results/'+'['+str(time_str)+']原始SVM.csv', index=False)


    '''返回每次运行的结果'''
    results_back = [data_set,noise_rate,pr1, avg_score,tm1, tm2]
    return results_back


def crf_nfl_svm(data):
    Row = data.shape[0]     #行数，样本数
    Column = data.shape[1]-1      #特征数
    
    print(Row,Column)
    
    df = pd.DataFrame(data)     #转为dataframe
    
    dfNum = int(0.8 * Row)
    trainNum = int(12 / 15 * dfNum)
    df1 = df.iloc[:dfNum, :]
    train = df1.iloc[:trainNum, :].values
    Validationdata = df1.iloc[trainNum:, :].values
    test = df.iloc[dfNum:, :].values
    
    _ntree = 100
    _niMax = int(Column / 2) + 1
    pr1, pr2, Ntree, Ni = CRF_NFL_SVM_XH.crfnfl_all(train, test, Validationdata, _ntree, _niMax)
    results = []
    results.append([pr1, pr2, Ntree, Ni])
    columns = ['原始精度', '去噪后精度', '最优Ntree', '最优Ni']
    d1 = pd.DataFrame(results, columns=columns)
    print(d1)
    
    curtime = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curtime,'%Y-%m-%d=%H-%M-%S')
    d1.to_csv(r'results/'+'['+str(time_str)+']CRF_NFL_SVM_.csv', index=False)


def crf_nfl_ersvm(data):
    Row = data.shape[0]     #行数，样本数
    Column = data.shape[1]-1      #特征数
    
    print(Row,Column)
    df = pd.DataFrame(data)     #转为dataframe

    dfNum = int(0.8 * Row)
    trainNum = int(12 / 15 * dfNum)
    df1 = df.iloc[:dfNum, :]
    train = df1.iloc[:trainNum, :].values       
    Validationdata = df1.iloc[trainNum:, :].values
    test = df.iloc[dfNum:, :].values
    
    returnSvm, returnCRFNFL_SVM, Ntree, Ni, Vote = CRF_NFL_ersvm_XH.crfnfl_all(
        train, test, Validationdata, _ntree=100,
        _niMax=int(Column / 2) + 1,
        _data_name=dn, _noise_rate=nr)
    a.append([dn, nr, returnSvm, returnCRFNFL_SVM, Ntree, Ni, Vote])


#添加翻转噪声
def tran(y_array, noise_rate=0):
    y_list = y_array.tolist()
    # print(y_list)
    l0 = []
    l1 = []
    for i in range(len(y_list)):
        if y_list[i][0] == 0:
            l0.append(i)        #添加索引
        elif y_list[i][0] == 1:
            l1.append(i)

    # print('l0',len(l0),'\tl1',len(l1))
    num0 = int(len(l0) * noise_rate)  # 每类标签需要翻转的数量
    num1 = int(len(l1) * noise_rate)

    # print('num0',num0)
    n0 = random.sample(l0, num0)        
    n1 = random.sample(l1, num1)

    # print('n0',n0)
    for a0 in n0:  # 置换标签
        y_list[a0][0] = 1
    for a1 in n1:
        y_list[a1][0] = 0

    # print(y_list)
    y_array = np.array(y_list)
    # print(np.array(y_list))       
    return y_array      #ndarray, (61,1)




def wf_svm_noise():
    '''
        用经典svm跑wf数据集，噪声率5~35.
        并保存验证精度和时间的变化
    '''
    
    X,Y = read_data(path=r'./水流量.csv',choose=1) 
    Y = Y.reshape(61,1)
    noise = [round(x*0.05,2) for x in range(1,8,1)]
    result_svm = []
    for noise_rate in noise:
        Y = tran(y_array=Y,noise_rate=noise_rate)   #添加翻转噪声
        data = np.hstack((Y,X))     #第一列是标签
        results_back = svm(raw_data=data,data_set='waterflow',noise_rate = noise_rate)
        result_svm.append(results_back)

    # print(result_svm)

    data = pd.read_excel(r'实验数据/SVM_不同噪声.xls')
    # print(data)

    columnss = data.columns
    new_data = pd.concat(
        [data,
        pd.DataFrame(result_svm,columns=columnss,index=range(98,105))])

    print(new_data)
    curtime = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curtime,'%Y-%m-%d=%H-%M-%S')
    new_data.to_excel('results/'+'['+str(time_str)+']SVM_不同噪声.xls', index=False)


def main():
    X,Y = read_data(path=r'./水流量.csv',choose=1) 
    Y = Y.reshape(61,1)  
    data = np.hstack((Y,X))     #第一列是标签


    # svm(raw_data=data)
    # crf_nfl_svm(data)


    



if __name__ == "__main__":
    # main()
    wf_svm_noise()