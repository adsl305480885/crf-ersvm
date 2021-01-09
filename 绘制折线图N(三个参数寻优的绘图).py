import pandas as pd
import matplotlib.pyplot as plt

# 设置输出的图片大小
figsize =8, 6
figure, ax = plt.subplots(figsize=figsize)

df=pd.read_excel('E:/123/test/3种方法表格/4.24\CRF_SVM_参数寻优整理结果-行列转置.xls',sheet_name='Ntree')
# df=pd.read_excel('E:/123/test/3种方法表格/4.24\CRF_SVM_参数寻优整理结果-行列转置.xls',sheet_name='NI')
# df=pd.read_excel('E:/123/test/3种方法表格/4.24\CRF_SVM_参数寻优整理结果-行列转置.xls',sheet_name='Voting Ratio')
x=df['Ntree'] #设置x轴数据
# x=df['NI']
# x=df['Voting Ratio']
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


#开始绘图，y1,y2,y3,y4分别代表4根折线
plt.plot(x,y1, color='orange', linestyle='-',linewidth=1.75)
plt.plot(x,y2, color='blue', linestyle='-',linewidth=1.75)
plt.plot(x,y3, color='black', linestyle='-',linewidth=1.75)
plt.plot(x,y4, color='blueviolet', linestyle='-',linewidth=1.75)
plt.plot(x,y5, color='deepskyblue', linestyle='-',linewidth=1.75)
plt.plot(x,y6, color='red', linestyle='-',linewidth=1.75)
plt.plot(x,y7, color='hotpink', linestyle='-',linewidth=1.75)
plt.plot(x,y8, color='green', linestyle='-',linewidth=1.75)
plt.plot(x,y9, color='lime', linestyle='-',linewidth=1.75)
plt.plot(x,y10, color='brown', linestyle='-',linewidth=1.75)

plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()

font2 = {'size':'15'}

# x_values=list(range(2,20,2))
# plt.plot(x_values)
#设置x和y轴的标签
plt.xlabel(u'Ntree',font2)
# plt.xlabel(u'NI',font2)
# plt.xlabel(u'VP(%)',font2)
plt.ylabel(u'accuracy',font2)
# plt.ylabel(u'runtime(s)')

# plt.legend() #在图的右上角加上图例，用于说明图中各种颜色的线代表哪种
# legend()函数中：
# labels用于设置每根折线代表的含义
# loc用于设置图例的位置，位置如下：
#      'best'         : 0, (only implemented for axes legends)(自适应方式)
#      'upper right'  : 1,
#      'upper left'   : 2,
#      'lower left'   : 3,
#      'lower right'  : 4,
#      'right'        : 5,
#      'center left'  : 6,
#      'center right' : 7,
#      'lower center' : 8,
#      'upper center' : 9,
#      'center'       : 10,
plt.show()

