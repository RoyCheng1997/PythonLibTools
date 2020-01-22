# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:11:34 2019
Linear Algebra @ numpy,linalg
@author: RoyCheng
@refer: https://docs.scipy.org/doc/numpy/reference/routines.linalg.html
"""
import numpy as np
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
# Compute the (multiplicative) inverse of a matrix.
np.linalg.inv(mas)


# Solving equations---------------
# Solve a linear matrix equation, or system of linear scalar equations.
np.linalg.solve(mas,np.asarray([3,4]))
# Return the least-squares solution to a linear matrix equation.
np.linalg.lstsq(mas,np.asarray([3,4]))
