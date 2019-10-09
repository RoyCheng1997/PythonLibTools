# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:52:06 2017

@author: RoyCheng
"""
#Numpy Codes
import numpy as np
from numpy.linalg import inv,qr

#创建ndarray#########################################################
data1 = [[1,2,3,4],[5,6,7,8]]
arr1 = np.array(data1,dtype=np.int32)
float_arr1 = arr1.astype(np.float64)#输出强制转换ndarry
zero_array = np.zeros((3,6)) #创建3X6全零矩阵
empty_array = np.empty((2,3,2)) #Random numbers
int_array = np.arange(10)
interval_array = np.arange(-5,5,0.01)
xs,ys = np.meshgrid(interval_array,interval_array)#由两个一维获取一个二维
#Other functions:ones ones_like zeros_like

#生成随机数######
data = np.random.randn(7,4) #随机数矩阵 正态分布
samples = np.random.normal(loc=2,scale=1,size=(4,4))#生成均值为2方差为1的4X4方阵
#np.random.randint() #给定上下线选取随机整数
#np.random.binomial() #二项分布
#np.random.beta() #Beta分布
#np.random.chisquare() #卡方分布
#np.random.gamma() #Gamma分布
#np.random.uniform() #0,1均匀分布
#shuffle对序列就地随机排列 permutation 返回序列随机排列或随机排列的范围


#索引切片###############################################################
arr = np.arange(10)
arr[5]
arr[5:8] #5 6 7
arr_copy = arr.copy() #拷贝副本
arr[5:8]=12 #赋值 广播
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[2] #行索引
arr2d[0][2]
arr2d[0,2]#行列索引 直接定位到值
arr2d[:2] #切片0 1行
arr2d[:2,1:] #0,1行 1,2列
arr2d[:,:1]
#Bool索引
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data[names=='Bob']#Bool值用于索引 要求轴长度一致
data[-(names=='Bob')]#反向索引
data[(names=='Bob')|(names=='Will')]
data[data<0]=0#赋值 广播
data[names!='Joe']=7#赋值 广播
#花式索引
arr=np.empty((8,4))
for i in range(8):
    arr[i]=i
arr[[4,3,0,6]] #按照指定顺序选择行子集
arr[[-3,-5,-7]]#负数表示从末尾开始选
arr=np.arange(32).reshape((8,4)) #重排列
arr[[1,5,7,2],[0,3,1,2]]#输出(1,0)(5,3)(7,1)(2,2)
arr[[1,5,7,2]][:,[0,3,1,2]] #按顺序选取行列重新组成矩阵
arr[np.ix_([1,5,7,2],[0,3,1,2])] #同上


#Calculation################################################################
#math
arr1*arr1
arr1*10
arr1+arr1
1/arr1
arr1**0.5
np.sqrt(arr)
np.exp(arr)
int_arr,float_arr=np.modf(arr)#分别接受小数和整数部分
arr[(int(0.05*len(arr)))]#5%分位数
#Others(1d):abs绝对值 square平方 log sign正负符号 ceil大于等于该值的最小整数 floor
#rint四舍五入保留整数 isnan是否为空值 isfinite isinf是否有穷 cos sin.... 
#转置与轴兑换
arr = np.arange(15).reshape((3,5))
arr.T #转置
arr.sort(1)#按轴排序
np.dot(arr.T,arr)#内积
arr=np.arange(16).reshape((2,2,4))#三维数组
arr.transpose((1,0,2))#高维数组需由轴编号组成的元组进行转置
arr.swapaxes(1,2)
#Others(2d):add subtract multiply power次方 copysign将第二个数组的值的符号复制给第一个数组
#greater greater_equal 比较运算 产生布尔值

#linalg线性代数#########
X=np.random.randn(5,5)
np.diag(X)#返回对角线元素或将一维数组转换成方阵
np.trace(X)#对角线元素和 迹
np.linalg.det(X)#行列式
np.linalg.eig(X)#计算特征值与特征向量
np.linalg.inv(X)#方阵的逆
np.linalg.qr(X)#QR分解
np.linalg.svd(X)#奇异值分解
#solve解线性方程Ax=b Istsq 计算AX=b最小二乘解


#where矢量化:数组代替循环#####################################################
xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr=np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True,False,True,True,False])
result = np.where(cond,xarr,yarr) #若条件满足 则该位置取x 否则取 y
result = [(x if c else y) for x,y,c in zip(xarr,yarr,cond)] #若条件满足 则该位置取x 否则取 y
arr=np.random.randn(4,4)
np.where(arr>0,2,arr)#只将正值设为2
#循环嵌套

result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)        
    else:
        result.append(3)        
        
result1 = np.where(cond1&cond2,0,np.where(cond1,1,np.where(cond2,2,3)))


#统计########################################################################
arr = np.random.randn(5,4)
bools = np.array([False,False,True,False])
arr.mean()
arr.sum()
(arr>0).sum()#正值的数量 bool值强制转化
#std var min max argmin最小索引 cumsum所有元素累积和 cumprod所有元素累积积
arr.cumsum(0)
arr.cumprod(1)
bools.any() #是否存在true
bools.all() #是否都是true
np.unique(names) #返回唯一的值
arr1 = np.array([6,0,0,3,2,5,6])
np.in1d(arr1,[2,3,6])#判断arr1中的元素是否在后面数组中存在
#intersect1d交集 union1d并集 setdiff1d差集 setxor1d对称差
np.allclose(arr1,arr2)#检查两个对象是否包含相同数据

#文件############################################################################
np.save('some_array',arr)
arr = np.load('some_array.npy')
arr = np.loadtxt('array_ex.txt',delimiter=',')




