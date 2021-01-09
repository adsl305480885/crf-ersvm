import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel('E:/123/test/3种方法表格/4.24/三种方法精度比较(精选).xls')
x=df['简称'] #设置x轴数据
y1=df['Classical-SVM']#设置y轴数据
y2=df['CRF-NFL-SVM']#设置y轴数据
y3=df['CRF-ERSVM']
# y3=df['CRF-NFL-SVM-Improve']
# y4=df['CRF-ERSVM-Improve']


# 开始绘图，y1,y2,y3,y4分别代表4根折线
plt.plot(x,y1, color='green')
plt.plot(x,y2, color='blue')
plt.plot(x,y3, color='RED')
# plt.plot(x,y3, color='blue')
# plt.plot(x,y4, color='red')

#设置x和y轴的标签
# plt.xlabel(u'noise-rate(%)')
plt.xlabel(u'datasets')
#plt.ylabel(u'runtime(s)')
# plt.ylabel(u'accuracy improvement(%)')
plt.ylabel(u'accuracy')

plt.legend() #在图的右上角加上图例，用于说明图中各种颜色的线代表哪种
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

