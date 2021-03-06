# -*- coding:utf-8 -*-
__author__ = 'HeatDeath'
__date__ = '2017/7/17 15:08'

from numpy import *

# 使用 random.rand() 命令生成一个 4*4 的随机数组
generate_random_array = random.rand(4, 4)
print(generate_random_array)
print(type(generate_random_array))

# # random.rand() 第一个参数标识行数，第二个参数标识列数
# test_array = random.rand(5, 6)
# print(test_array)

# --------------------------------------------------
print('---------------------------------------------')
# --------------------------------------------------

# mat() 函数将 数组 array 转化为 矩阵 matrix
randMat = mat(generate_random_array)
print(randMat)
print(type(randMat))

# --------------------------------------------------
print('---------------------------------------------')
# --------------------------------------------------

# .I 操作符实现了 矩阵 求逆 运算
invRandMat = randMat.I
print(invRandMat)
print(type(invRandMat))

# --------------------------------------------------
print('---------------------------------------------')
# --------------------------------------------------

# 矩阵 * 逆矩阵 = 单位矩阵
mul_result = randMat*invRandMat
print(mul_result)

# --------------------------------------------------
print('---------------------------------------------')
# --------------------------------------------------

# eye() 函数可以创建指定大小的单位矩阵
sub_result = mul_result - eye(4)
print(sub_result)

# --------------------------------------------------
print('---------------------------------------------')
# --------------------------------------------------


