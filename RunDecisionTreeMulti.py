#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: RunDecisionTreeMulti.py\n
@time: 2019-10-12 15:27\n
@desc: 决策树多元分类（DecisionTree Multi Class Classification）——预测在不同条件下（Elevation-海拔、Aspect-方位、Slope-斜率、水源的垂直距离、荒野分类、水源的水平距离、土壤分类等）适合种植的植被，并通过训练评估找出最佳参数组合，提高预测准确度
'''
import os, sys
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics


os.environ['SPARK_HOME'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6'
os.environ['JAVA_HOME'] = 'D:\ProgramFiles\Java\jdk1.8.0_131'
os.environ['HADOOP_HOME'] = 'D:\ProgramFiles\hadoop\hadoop-2.6.4'
os.environ['PYSPARK_PYTHON'] = 'D:\ProgramFiles\Python364\python.exe'  # 在dos命令行下执行时需要，在IDE中运行配置到Run Configurations中即可，只能使用Python364中的python.exe，而Anaconda3中的python.exe报错：Cannot run program "?D:\ProgramFiles\Anaconda3\python.exe": CreateProcess error=2,

def CreateSparkContext():
    '''
    创建SparkContext对象
    :return:
    '''
    sparkConf = SparkConf().setAppName('RunDecisionTreeMulti').set('spark.ui.showConsoleProgress', 'false')
    sc = SparkContext(conf=sparkConf)
    print('master=' + sc.master)
    SetLogger(sc)
    SetPath(sc)
    return sc

def SetLogger(sc):
    '''
    设置日志级别
    :param sc:
    :return:
    '''
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger('org').setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger('akka').setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

def SetPath(sc):
    '''
    设置文件系统路径
    :param sc:
    :return:
    '''
    global Path
    if sc.master[0:5] == 'local':
        Path = u'file:/E:/SparkPythonWorkspace/PySparkDemo/'
    else:
        Path = u'hdfs://master:9000/user/root/'

def PrepareData(sc):
    '''
    准备数据
    :param sc:
    :return: (trainData, validationData, testData)
    '''
    print('======================= 准备数据 =======================')
    # ----------------------------- 1. 导入并转换数据 -----------------------------
    print('========== [PrepareData] >>>> 开始导入 covtype.data 数据....')
    rawData = sc.textFile(Path + u'data/covertype/covtype.data')
    print('========== [PrepareData] >>>> 共计：' + str(rawData.count()) + ' 项')
    lines = rawData.map(lambda x: x.split(','))
    # print('========== [PrepareData] >>>> 共计：' + str(lines.count()) + ' 项')
    # ----------------------------- 2. 建立训练评估所需数据RDD[LabeledPoint] -----------------------------
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, -1)))
    # ----------------------------- 3. 以随机方式将数据分为3个部分并返回 -----------------------------
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print('========== [PrepareData] >>>> 将数据以随机方式差分为三个部分：trainData: ' + str(trainData.count()) + ' 项, validationData: ' + str(validationData.count()) + ' 项, testData: ' + str(testData.count()) + ' 项')
    # ----------------------------- 4. 返回元组数据 -----------------------------
    return (trainData, validationData, testData)

def extract_label(record):
    '''
    提取数据中的label特征字段
    :param record:
    :return:
    '''
    label = (record[-1])
    return float(label) - 1

def extract_features(record, featureEnd):
    '''
    提取数据的feature特征字段
    :param record:
    :param featureEnd:
    :return:
    '''
    numericalFeatures = [convert_float(field) for field in record[0:featureEnd]]
    return numericalFeatures

def convert_float(x):
    '''
    将内容转换为float类型数据
    :param x:
    :return:
    '''
    return (0 if x == '?' else float(x))

def trainEvaluateModel(trainData, validationData, impurityParm, maxDepthParm, maxBinsParm):
    '''
    训练模型时会输入不同的参数。其中，DecisionTree参数有impurity、maxDepth、maxBins等的值都会影响准确率以及训练所需的时间。
    我们以图表显示这些参数值、准确率与训练所需的时间。
    我们每次只会评估单个参数的不同值，例如评估maxDepth参数的不同值[3, 5, 10, 15, 20, 25]，执行步骤如下：
    （1）用DecisionTree.trainClassifier进行训练传入trainData与单个参数的不同数值；
    （2）建立模型后，用validationData评估模型的accuracy准确率；
    （3）训练与评估模型重复执行多次，产生多个参数项的accuracy与运行时间，并存储于metricsRDD中；
    （4）全部执行完成后，将metricsRDD转换为Pandas DataFrame；
    （5）Pandas DataFrame可绘制accuracy与运行时间图表，用于显示不同参数的准确率与执行时间的关系。
    :param trainData:
    :param validationData:
    :param impurityParm:
    :param maxDepthParm:
    :param maxBinsParm:
    :return:
    '''
    print('======================= 训练评估模型 =======================')
    startTime = time()
    model = DecisionTree.trainClassifier(trainData, numClasses=7, categoricalFeaturesInfo={}, impurity=impurityParm, maxDepth=maxDepthParm, maxBins=maxBinsParm)
    accuracy = evaluateModel(model, validationData)
    duration = time() - startTime
    print('========== [trainEvaluateModel] >>>> 训练评估模型：使用参数：impurity=' + str(impurityParm) + ', maxDepth=' + str(maxDepthParm) + ', maxBins=' + str(maxBinsParm) + '\n' +
          '\t\t==>> 所需时间=' + str(duration) + ', 结果accuracy=' + str(accuracy))
    return (accuracy, duration, impurityParm, maxDepthParm, maxBinsParm, model)


def evaluateModel(model, validationData):
    '''
    使用accuracy评估模型的准确率
    :param model:
    :param validationData:
    :return:
    '''
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = MulticlassMetrics(scoreAndLabels)
    accuracy = metrics.accuracy
    return (accuracy)

def evalParameter(trainData, validationData, evalParm, impurityList, maxDepthList, maxBinsList):
    '''
    评估影响模型准确性的不同参数
    # 调用示例：
    evalParameter(trainData, validationData, 'maxDepth',
                impurityList=['gini'],
                maxDepthList=[3, 5, 10, 15, 20, 25],
                maxBinsList=[10])

    :param trainData:
    :param validationData:
    :param evalParm:
    :param impurityList: ['gini', 'entropy']
    :param maxDepthList:
    :param maxBinsList:
    :return:
    '''
    print('======================= 评估模型参数(' + evalParm + ') =======================')
    # 训练评估参数
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 设置当前评估的参数
    if evalParm == 'impurity':
        indexList = impurityList[:]
    elif evalParm == 'maxDepth':
        indexList = maxDepthList[:]
    elif evalParm == 'maxBins':
        indexList = maxBinsList[:]
    # 转换为Pandas DataFrame
    df = pd.DataFrame(metrics, index=indexList, columns=['accuracy', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    # 显示图形
    showChart(df, evalParm, 'accuracy', 'duration', 0.6, 1.0)

def showChart(df, evalParm, barData, lineData, yMin, yMax):
    '''
    用Matplotlib绘制训练评估结果图形
    # 调用示例：showChart(showDataTable(trainData, validationData, ...), 'impurity', 'accuracy', 'duration', 0.5, 0.7)
    :param df:
    :param evalParm:
    :param barData:
    :param lineData:
    :param yMin:
    :param yMax:
    :return:
    '''
    # 绘制直方图
    ax = df[barData].plot(kind='bar', title=evalParm, figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(evalParm, fontsize=12)
    ax.set_ylim([yMin, yMax])
    ax.set_ylabel(barData, fontsize=12)
    # 绘制折线图
    ax2 = ax.twinx()
    ax2.plot(df[lineData].values, linestyle='-', marker='o', linewidth=2.0, color='r')
    plt.show()  # 绘制图形

def showDataTable(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    '''
    使用图表显示训练评估结果
    :param trainData:
    :param validationData:
    :return:
    '''
    # impurityList = ['gini', 'entropy']
    # maxDepthList = [10]
    # maxBinsList = [10]
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    indexList = impurityList
    df = pd.DataFrame(metrics, index=indexList, columns=['accuracy', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    return df

def PredictData(sc, model):
    '''
    对各个因素控制的因素森林环境是否生长植被进行预测的函数
    :param sc:
    :param model:
    :return:
    '''
    print('======================= 预测数据 =======================')
    print('========== [PredictData] >>>> 开始导入 covtype.data 数据....')
    rawData = sc.textFile(Path + u'data/covertype/covtype.data')
    lines = rawData.map(lambda x: x.split(','))
    print('========== [PredictData] >>>> 共计：' + str(rawData.count()) + ' 项')
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, -1)))
    for lp in labelpointRDD.take(10):
        predict = model.predict(lp.features)
        label = lp.label
        features = lp.features
        result = (' 正确 ' if (label == predict) else '错误')
        print('========== [PredictData] >>>> 土地条件：海拔：' + str(features[0]) +
                '，方位：' + str(features[1]) +
                '，斜率：' + str(features[2]) +
                '，水源垂直距离：' + str(features[3]) +
                '，水源水平距离：' + str(features[4]) +
                '，9点时阴影：' + str(features[5]) + '\n' +
                '\t\t....==>> 预测：' + str(predict) + '，实际：' + str(label) + '，结果：' + result + '\n')

def parameterEval(trainData, validationData):
    '''
    评估'impurity', 'maxDepth', 'maxBins'等参数
    :param trainData:
    :param validationData:
    :return:
    '''
    # 评估 impurity 参数
    evalParameter(trainData, validationData, 'impurity',
                  impurityList=['gini', 'entropy'],
                  maxDepthList=[10],
                  maxBinsList=[10])
    # 评估 maxDepth 参数
    evalParameter(trainData, validationData, 'maxDepth',
                    impurityList=['gini'],
                    maxDepthList = [3, 5, 10, 15, 20, 25],
                    maxBinsList = [10])
    # 评估 maxBins 参数
    evalParameter(trainData, validationData, 'maxBins',
                  impurityList=['gini'],
                  maxDepthList=[10],
                  maxBinsList=[3, 5, 10, 50, 100, 200])

def evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    '''
    训练评估所有参数，找出最好的参数组合
    # 调用示例：
    bestModel = evalAllParameter(trainData, validationData,
            ['gini', 'entropy'],
            [3, 5, 10, 15, 20, 25],
            [3, 5, 10, 50, 100, 200])

    :param trainData:
    :param validationData:
    :param impurityList:
    :param maxDepthList:
    :param maxBinsList:
    :return:
    '''
    print('======================= 训练评估所有参数，找出最好的参数组合 =======================')
    # for循环训练评估所有参数组合
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 找出accuracy最大的参数组合
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter = Smetrics[0]
    # 显示调校后最佳参数组合
    print('========== [evalAllParameter] >>>> 调校后最佳参数组合：impurity=' + str(bestParameter[2]) + ', maxDepth=' + str(bestParameter[3]) + ', maxBins=' + str(bestParameter[4]) + '\n' +
          '\t\t==>> 结果accuracy=' + str(bestParameter[0]))
    # 返回最佳模型
    return bestParameter[5]

if __name__ == '__main__':
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ Start RunDecisionTreeMulti @@@@@@@@@@@@@@@@@@@@@@@@@')
    sc = CreateSparkContext()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据准备阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (trainData, validationData, testData) = PrepareData(sc)
    # 为了提高运行效率暂将数据保存在内存中
    trainData.persist()
    validationData.persist()
    testData.persist()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 训练评估阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (accuracy, duration, impurityParm, maxDepthParm, maxBinsParm, model) = trainEvaluateModel(trainData, validationData, 'entropy', 15, 50)
    if (len(sys.argv) == 2) and (sys.argv[1] == '-e'):
        parameterEval(trainData, validationData)
    elif (len(sys.argv) == 2) and (sys.argv[1] == '-a'):
        print('@@@@@@@@-------- 所有参数训练评估找出最好的参数组合 --------@@@@@@@@')
        model = evalAllParameter(trainData, validationData,
                                ['gini', 'entropy'],
                                [3, 5, 10, 15, 20, 25],
                                [3, 5, 10, 50, 100, 200])
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 测试阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    acc = evaluateModel(model, testData)
    print('@@@@@@@@============== [Main] >>>> 使用 testData 测试最佳模型，结果accuracy=' + str(acc))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 预测数据 @@@@@@@@@@@@@@@@@@@@@@@@@')
    PredictData(sc, model)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 查看分类规则 @@@@@@@@@@@@@@@@@@@@@@@@@')
    # print('@@@@@@@@============== [Main] >>>> Print Debug Info: \n')
    print(model.toDebugString())