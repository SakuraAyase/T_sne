from sklearn.datasets import *
import matplotlib.pyplot as plt
from numpy import *
from PIL import Image
import os
import time
import scipy as sp

import pandas as pd
import csv
import copy

data_set=load_iris()
features=data_set["data"]
x=features
y=data_set["target"]
print(features)
print(len(features))
x=array(features[:,:2])
print(x)
plt.plot(x.T[0][1:50],x.T[1][1:50],'ro',color='r')
plt.plot(x.T[0][50:100],x.T[1][50:100],'ro',color='g')
plt.plot(x.T[0][100:150],x.T[1][100:150],'ro',color='b')
plt.show()
z=array(random.randn(len(x),2))*0.1

def dis_x(vec1,vec2):
    difference=vec1-vec2
    return float(exp(-(dot(difference,difference.T))**0.5))
def dis_z(vec1,vec2):
    difference=vec1-vec2
    dis=dot(difference,difference.T)**0.5
    return float(1/(1+dis))

def cost(P,Q):
    if(P>Q):
        n=log(P/(Q+0.0000001))
    else:
        n=log(Q/(P+0.0000001))
    return P*n
def dcost(P,Q):
    if(P>Q):
        return -P / (Q + 0.0000001)
    else:
        return P/(Q+0.0000001)

Probability=array(eye(len(x),len(x)))
"""for i in range(len(x)):
    x[i]=x[i]/((dot(x[i],x[i].T))**0.5)"""
for i in range(len(x)):
    for j in range(len(x)):
        Probability[i][j]=dis_x(x[i],x[j])

for i in range(len(x)):
    Probability[i]=Probability[i]/sum(Probability[i])
print(shape(Probability))
print(Probability[0])
alpha=0.8
for n in range(1000):
    Pro_z=array(eye(len(z),len(z)))
    for i in range(len(z)):
        for j in range(len(z)):
            Pro_z[i][j]=dis_z(z[i],z[j])
    for i in range(len(z)):
        Pro_z[i]=Pro_z[i]/sum(Pro_z[i])
    #print(Pro_z[0])

    cost1=0.0
    for i in range(len(z)):
        for j in range(len(z)):
            cost1+=cost(Probability[i][j],Pro_z[i][j])
    print(cost1)

    dz=[]
    for i in range(len(z)):
        dz.append([])
    for i in range(len(z)):
        for j in range(len(z)):
            dL=dcost(Probability[i][j],Pro_z[i][j])
            dis = dot((z[i]-z[j]), (z[i]-z[j]).T) ** 0.5+0.1
            dZ=-1*((dis_z(z[i],z[j])**2))/dis*(z[i]-z[j])
            dz[i].append(dZ*dL/len(z))

    dz1=[]
    for i in range(len(z)):
        temp=array([1.0,0])
        for j in range(len(z)):
            temp+=dz[i][j]
        dz1.append(temp)
    for i in range(len(z)):
        z[i]=z[i]-alpha*dz1[i]

    z1=z[0:50].T
    z2=z[50:100].T
    z3=z[100:150].T
    """z_temp=z.T
    plt.plot(z1[0], z1[1], 'ro',color='r')
    plt.plot(z2[0], z2[1], 'ro',color='b')
    plt.plot(z3[0], z3[1], 'ro',color='g')"""
    ##print(z_out)
    plt.plot(z1[0], z1[1], 'ro',color='r')
    plt.plot(z2[0], z2[1], 'ro', color='b')
    plt.plot(z3[0], z3[1], 'ro', color='g')
    plt.savefig("C:\\Users\\myfamily\\Desktop\\新建文件夹\\loop"+str(n))
    plt.pause(1)
    plt.close()
    #plt.show()



"""plt.plot(z1[0], z1[1], marker="P", color='b', label='data1')
plt.plot(z2[0], z2[1], marker="o", color='k', label='data2')
plt.plot(z3[0], z3[1], marker="*", color='r', label='data3')"""







