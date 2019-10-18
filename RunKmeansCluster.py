#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: RunKmeansCluster.py.py\n
@time: 2019-10-18 10:17\n
@desc: k-means 聚类算法——使用鸢尾花数据集进行实战
    数据源地址：https://archive.ics.uci.edu/ml/datasets/Iris
'''
import numpy as np
import matplotlib.pyplot as plt
# from pyspark.ml.clustering import KMeans
from sklearn.cluster import KMeans
import seaborn as sns

def loadDataset():
    '''
    加载数据集
    seaborn.load_dataset(name) 方法会从：https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{name}.csv 加载数据
    :return:
    '''
    data = sns.load_dataset('iris', cache=False)  # 不加cache=False会报错：IndexError: single positional indexer is out-of-bounds
    return data

def observeData(data):
    '''
    观察数据
    :param data:
    :return:
    '''
    print(str(data.head()))  # 默认显示前5条数据
    pg = sns.pairplot(data, hue='species')  # 返回PairGrid对象
    plt.show()  # 显示图形

def trainModel(data, clusterNum):
    '''
    使用KMeans对数据进行聚类
    '''
    # max_iter表示EM算法迭代次数，n_init表示K-means算法迭代次数，algorithm="full"表示使用EM算法。
    model = KMeans(n_clusters=clusterNum, max_iter=100, n_init=10, algorithm='full')
    model.fit(data)
    return model


def computeSSE(model, data):
    '''
    计算聚类结果的误差平方和
    '''
    wdist = model.transform(data).min(axis=1)
    sse = np.sum(wdist ** 2)
    return sse

def evalClusterCounts(data):
    '''
    评估聚类个数
    :param data:
    :return:
    '''
    # 将 sepal_length,sepal_width,petal_length,petal_width 进行排列组合
    col = [['petal_width', 'sepal_length'], ['petal_width', 'petal_length'], ['petal_width', 'sepal_width'], \
           ['sepal_length', 'petal_length'], ['sepal_length', 'sepal_width'], ['petal_length', 'sepal_width']]

    for i in range(6):
        fig = plt.figure(figsize=(8, 8), dpi=80)
        ax = fig.add_subplot(3, 2, i + 1)
        sse = []
        for j in range(2, 6):
            model = trainModel(data[col[i]], j)
            sse.append(computeSSE(model, data[col[i]]))
        ax.plot(range(2, 6), sse, 'k--', marker="o", markerfacecolor="r", markeredgecolor="k")
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        title = "clusterNum of %s and %s" % (col[i][0], col[i][1])
        ax.title.set_text(title)
        plt.show()

def showClusterResult(clusterParam, clusterNum):
    '''
    选择一组指标进行可视化聚类，观察结果
    :param clusterParam: 例如：['petal_width', 'petal_length']
    :param clusterNum: 聚类个数由evalClusterCounts()方法评估得出
    :return:
    '''
    petal_data = data[clusterParam]
    model = trainModel(petal_data, clusterNum)  # 此处的聚类个数由evalClusterCounts()方法评估得出
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    colors = ["r", "b", "g"]
    # ax.scatter(petal_data.petal_width, petal_data.petal_length, c=[colors[i] for i in model.labels_], marker="o", alpha=0.8)
    ax.scatter(petal_data[clusterParam[0]], petal_data[clusterParam[1]], c=[colors[i] for i in model.labels_], marker="o", alpha=0.8)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="*", c=colors, edgecolors="white", s=700., linewidths=2)
    # yLen = petal_data.petal_length.max() - petal_data.petal_length.min()
    yLen = petal_data[clusterParam[1]].max() - petal_data[clusterParam[1]].min()
    # xLen = petal_data.petal_width.max() - petal_data.petal_width.min()
    xLen = petal_data[clusterParam[0]].max() - petal_data[clusterParam[0]].min()
    lens = max(yLen + 1, xLen + 1) / 2.
    # ax.set_xlim(petal_data.petal_width.mean() - lens, petal_data.petal_width.mean() + lens)
    ax.set_xlim(petal_data[clusterParam[0]].mean() - lens, petal_data[clusterParam[0]].mean() + lens)
    # ax.set_ylim(petal_data.petal_length.mean() - lens, petal_data.petal_length.mean() + lens)
    ax.set_ylim(petal_data[clusterParam[1]].mean() - lens, petal_data[clusterParam[1]].mean() + lens)
    ax.set_ylabel(clusterParam[1])
    ax.set_xlabel(clusterParam[0])
    plt.show()


if __name__ == "__main__":
    data = loadDataset()
    # observeData(data)
    # evalClusterCounts(data)
    showClusterResult(['petal_width', 'petal_length'], 3)
    showClusterResult(['petal_width', 'sepal_width'], 3)
    showClusterResult(['petal_width', 'sepal_length'], 3)
    showClusterResult(['sepal_length', 'petal_length'], 3)
    showClusterResult(['sepal_length', 'sepal_width'], 3)
    showClusterResult(['petal_length', 'sepal_width'], 3)



