'''
Author: Zhou Hao
Date: 2021-01-06 22:28:16
LastEditors: Zhou Hao
LastEditTime: 2021-01-09 23:03:05
Description: file content
E-mail: 2294776770@qq.com
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



def main():
    # 设置输出的图片大小
    figsize =8, 6
    figure, ax = plt.subplots(figsize=figsize)


    df_1 = pd.read_excel(r'实验数据\SVM_noiserate-acc.xls',sheet_name='Sheet2')
    df_2 = pd.read_excel(r'实验数据\SVM_noiserate-time.xls',sheet_name='Sheet2')
    # print(df_1)
    for index,df in enumerate([df_1,df_2]):

        x=df['noise rate']
        y1=df['sonar']#设置y轴数据
        y2=df['votes']#设置y轴数据
        y3=df['clean1']
        y4=df['BC']
        y5=df['CA']
        y6=df['4class']
        y7=df['splice']
        y8=df['svmG3']
        y9=df['isolet']
        y10=df['svmG1']
        y11 = df['wf']


        #开始绘图，y1,y2,y3,y4分别代表4根折线
        plt.plot(x,y1, color='orange',linestyle='-',linewidth=1.75)
        plt.plot(x,y2, color='blue',linestyle='-',linewidth=1.75)
        plt.plot(x,y3, color='black',linestyle='-',linewidth=1.75)
        plt.plot(x,y4, color='blueviolet',linestyle='-',linewidth=1.75)
        plt.plot(x,y5, color='deepskyblue',linestyle='-',linewidth=1.75)
        plt.plot(x,y6, color='red',linestyle='-',linewidth=1.75)
        plt.plot(x,y7, color='hotpink',linestyle='-',linewidth=1.75)
        plt.plot(x,y8, color='green',linestyle='-',linewidth=1.75)
        plt.plot(x,y9, color='lime',linestyle='-',linewidth=1.75)
        plt.plot(x,y10, color='brown',linestyle='-',linewidth=1.75)
        plt.plot(x,y11, color='slategray',linestyle='-',linewidth=1.75)  #TODO color
        plt.tick_params(labelsize=15)

        font2 = {'size':'15'}

        plt.xlabel(u'Noise-rate(%)',font2)
        # plt.xlabel(u'NI')
        # plt.xlabel(u'VP')


        if index == 0: 
            plt.ylabel(u'accuracy',font2)
            #TODO,测试好了再保存，以免覆盖以前的文件
            # plt.savefig(fname=r'results/'+'svm_nr_accr'+'.pdf',
            #             format='pdf',
            #             bbox_inches='tight')
        elif index == 1:
            plt.ylabel(u'runtime(s)',font2)
            # plt.savefig(fname=r'results/'+'svm_nr_runtime'+'.pdf',
            #             format='pdf',
            #             bbox_inches='tight')
        plt.show()



def legend():
    # plt.scatter(1,1,label='minority class',
    #             c='tan', marker='o', s=25, )
    # plt.scatter(1,1,label='majority class',
    #             c='darkcyan', marker='o', s=25, )
    # plt.scatter(1,1,label='new samplers',
    #             c='red', s=35, marker='+')
    axes = plt.axes()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([]
    )
    


    df = pd.read_excel(r'实验数据\SVM_noiserate-acc.xls',sheet_name='Sheet2')
    x=df['noise rate']
    y1=df['sonar']#设置y轴数据
    y2=df['votes']#设置y轴数据
    y3=df['clean1']
    y4=df['BC']
    y5=df['CA']
    y6=df['4class']
    y7=df['splice']
    y8=df['svmG3']
    y9=df['isolet']
    y10=df['svmG1']
    y11 = df['wf']


    #开始绘图，y1,y2,y3,y4分别代表4根折线
    plt.plot(x,y1, color='orange',linestyle='-',linewidth=1.75,label='sonar')
    plt.plot(x,y2, color='blue',linestyle='-',linewidth=1.75,label='votes')
    plt.plot(x,y3, color='black',linestyle='-',linewidth=1.75,label='clean1')
    plt.plot(x,y4, color='blueviolet',linestyle='-',linewidth=1.75,label='BC')
    plt.plot(x,y5, color='deepskyblue',linestyle='-',linewidth=1.75,label='CA')
    plt.plot(x,y6, color='red',linestyle='-',linewidth=1.75,label='4class')
    plt.plot(x,y7, color='hotpink',linestyle='-',linewidth=1.75,label='splice')
    plt.plot(x,y8, color='green',linestyle='-',linewidth=1.75,label='svmG3')
    plt.plot(x,y9, color='lime',linestyle='-',linewidth=1.75,label='isolet')
    plt.plot(x,y10, color='brown',linestyle='-',linewidth=1.75,label='svmG1')
    plt.plot(x,y11, color='slategray',linestyle='-',linewidth=1.75,label='wf')  
    
    plt.legend(ncol=5,)
    # plt.legend(ncol=3,frameon=False)    
    # plt.savefig(fname='./pdf/'+'lengend'+'.pdf',format='pdf',bbox_inches='tight')        
    plt.show()




if __name__ == "__main__":
    # main()
    legend()

