# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:52:06 2017

@author: RoyCheng
"""
#Numpy Codes
import numpy as np
# %%
# Create ndarry =========================================
data = [[1,2,3,4],[5,6,7,8]]
arr = np.array(data,dtype=np.int32) # create from list
float_arr1 = arr.astype(np.float64) # transfer data type
zero_array = np.zeros((3,6)) # create 3X6 zero matrix
empty_array = np.empty((2,3,2)) # create 2X3X2 empty matrix with random number
int_array = np.arange(10) # range, arange, xrange
interval_array = np.arange(-5,5,0.01)
xs,ys = np.meshgrid(interval_array,interval_array)# mesh two 1-d array into a grid (2-d)
#Other functions:ones ones_like zeros_like

# Generate random number -----------------
data = np.random.randn(7,4) # 7X4 random matrix with normal distribution
samples = np.random.normal(loc=2,scale=1,size=(4,4))# 4X4 matrix with mean 2 variance 1
#np.random.randint() #给定上下线选取随机整数
#np.random.binomial() #二项分布
#np.random.beta() #Beta分布
#np.random.chisquare() #卡方分布
#np.random.gamma() #Gamma分布
#np.random.uniform() #0,1均匀分布
#shuffle对序列就地随机排列 permutation 返回序列随机排列或随机排列的范围


# %%
# Index & Slice ==================================
arr = np.arange(10)
arr[5]
arr[5:8] # 5 6 7
arr_copy = arr.copy() # copy
arr[5:8]=12 # assign, broadcast
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[2] # row
arr2d[0][2] == arr2d[0,2] # row,col
arr2d[:2] # slice row 0,1
arr2d[:2,1:] # row 0,1 col 1,2
arr2d[:,:1] # col 0
# Bool index ---------------------
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data[names=='Bob'] == data[(names=='Bob')] # Bool index, require same length
data[-(names=='Bob')] # reverse index
data[(names=='Bob')|(names=='Will')] # or, union
data[(names=='Bob')&(names=='Will')] # and, intersection, empty
data[data<0]=0 # assign, broadcast
data[names!='Joe']=7 # assign, broadcast
# Complex index --------------------
arr = np.empty((8,4))
for i in range(8):
    arr[i]=i # assign each row
arr[[4,3,0,6]] # select row according to specified order
arr[[-3,-5,-7]]# select row according to specified order (reverse direction)
arr = np.arange(32).reshape((8,4)) # reshape
arr[[1,5,7,2],[0,3,1,2]] # mix index with row,col (1,0)(5,3)(7,1)(2,2)
arr[[1,5,7,2]][:,[0,3,1,2]] # select rows and order cols
arr[np.ix_([1,5,7,2],[0,3,1,2])] #同上


# %%
# Calculation =======================================
# math -------------------------
arr = np.arange(10)
arr * arr == arr ** 2 #== np.square(arr) # square
arr ** 0.5 == np.sqrt(arr) # sqrt
arr * 2 == arr + arr
1/arr
np.exp(arr)
int_arr,float_arr=np.modf(arr) # integer & decimal part
arr[(int(0.05*len(arr)))] # 5% quantile, sample
np.quantile(arr,0.05) # quantile, uniform distribution
#Others(1d):abs绝对值 square平方 log sign正负符号 ceil大于等于该值的最小整数 floor
#rint四舍五入保留整数 isnan是否为空值 isfinite isinf是否有穷 cos sin.... 
# matrix-like operation -----------
arr = np.arange(15).reshape((3,5))
arr.T #转置
arr.sort(1)#按轴排序
np.dot(arr.T,arr)#内积
arr=np.arange(16).reshape((2,2,4))#三维数组
arr.transpose((1,0,2))#高维数组需由轴编号组成的元组进行转置
arr.swapaxes(1,2)
#Others(2d):add subtract multiply power次方 copysign将第二个数组的值的符号复制给第一个数组
#greater greater_equal 比较运算 产生布尔值


# %%
# where矢量化:array for loop ========================
xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True,False,True,True,False])
result = np.where(cond,xarr,yarr) # if cond true, choose x else choose y
result1 = [(x if c else y) for x,y,c in zip(xarr,yarr,cond)] #若条件满足 则该位置取x 否则取 y
result == result1
arr = np.random.randn(4,4)
arr = np.where(arr>0,2,arr) #只将正值设为2
# loop in loop
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
# better solution as:        
result1 = np.where(cond1&cond2,0,np.where(cond1,1,np.where(cond2,2,3)))


#%%
# Linear Algebra ===========================
# Matrix and vector products ---------------
vec1 = np.asarray([1,2,3,4])
vec2 = np.asarray([5,6,7,8])
ma1 = np.asarray([[1,2,3],[2,3,4]]) # 2*3 matrix
ma2 = np.asarray([[1,2],[2,3],[6,8]]) # 3*2 matrix
mas = np.asarray([[1,2],[3,4]]) # 2*2 matrix
# matrix multiply (shape must be matched)
np.matmul(ma1,ma2) # 2*2
np.matmul(ma2,ma1) # 3*3
# or use
np.dot(ma1,ma2) # 2*2
np.dot(ma2,ma1) # 3*3
# inner product/outer product
np.inner(vec1,vec2)
np.outer(vec1,vec2)
# power
mas**2
np.linalg.matrix_power(mas,2)

# Decompositions ---------------
# Cholesky decomposition(symetric positive definite)
np.linalg.cholesky(np.asarray([[2,1],[1,2]])) # A = R*R.T, R lower diagonal
# qr decomposition 
np.linalg.qr(np.asarray([[1,4],[1,2]])) # A=Q*U 
# svd decomposition
np.linalg.svd(np.asarray([[2,1],[1,2]])) # A = U*Diag*U.T (symetric positive definite)
np.linalg.svd(np.asarray([[1,4],[1,2]])) # A = U*Diag*V.T (symetric positive definite)

# Matrix eigenvalues ---------------
# Compute the eigenvalues and right eigenvectors of a square array.
np.linalg.eig(mas) # eigenvalues & eigenvectors
# Compute the eigenvalues of a general matrix.
np.linalg.eigvals(mas) # eigenvalues only

# Norms and other numbers ---------------
# Matrix or vector norm.
np.linalg.norm(mas) # norm
# Compute the determinant of an array.
np.linalg.det(mas)
# Return matrix rank of array using SVD method
np.linalg.matrix_rank(ma1)
# trace
np.trace(ma1)
# diagonal elements
np.diag(ma1)

# Compute the (multiplicative) inverse of a matrix.
np.linalg.inv(mas)

# Solving equations---------------
# Solve a linear matrix equation, or system of linear scalar equations.
np.linalg.solve(mas,np.asarray([3,4]))
# Return the least-squares solution to a linear matrix equation.
np.linalg.lstsq(mas,np.asarray([3,4]))


# %%
# Statistics ==============================
arr = np.random.randn(5,4)
bools = np.array([False,False,True,False])
arr.mean()
arr.sum()
(arr>0).sum() # 正值的数量 bool值强制转化
#std var min max argmin最小索引 cumsum所有元素累积和 cumprod所有元素累积积
arr.cumsum(0)
arr.cumprod(1)
bools.any() # 是否存在true
bools.all() # 是否都是true
np.unique(names) # 返回唯一的值
arr1 = np.array([6,0,0,3,2,5,6])
np.in1d(arr1,[2,3,6]) # 判断arr1中的元素是否在后面数组中存在
#intersect1d交集 union1d并集 setdiff1d差集 setxor1d对称差
np.allclose(arr1,arr2) # 检查两个对象是否包含相同数据


# %% 
# Files ======================================
np.save('some_array',arr)
arr = np.load('some_array.npy')
arr = np.loadtxt('array_ex.txt',delimiter=',')
