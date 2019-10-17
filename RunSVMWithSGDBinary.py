#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: RunSVMWithSGDBinary.py\n
@time: 2019-10-12 13:40\n
@desc: SVM(Support Vector Machine, 支持向量机)——一种监督式学习方法，使用StumbleUpon数据集，运用支持向量机SVM二元分类来预测网页是暂时性的还是长青的；并且通过训练评估找出最佳参数组合，提高预测准确度
    Stochastic Gradient Descent(SGD):随机梯度下降法
'''
import os, sys
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler

os.environ['SPARK_HOME'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6'
os.environ['JAVA_HOME'] = 'D:\ProgramFiles\Java\jdk1.8.0_131'
os.environ['HADOOP_HOME'] = 'D:\ProgramFiles\hadoop\hadoop-2.6.4'
os.environ['PYSPARK_PYTHON'] = 'D:\ProgramFiles\Python364\python.exe'  # 在dos命令行下执行时需要，在IDE中运行配置到Run Configurations中即可，只能使用Python364中的python.exe，而Anaconda3中的python.exe报错：Cannot run program "?D:\ProgramFiles\Anaconda3\python.exe": CreateProcess error=2,

def CreateSparkContext():
    '''
    创建SparkContext对象
    :return:
    '''
    sparkConf = SparkConf().setAppName('RunSVMWithSGDBinary').set('spark.ui.showConsoleProgress', 'false')
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
    :return: (trainData, validationData, testData, categoriesMap)
    '''
    print('======================= 准备数据 =======================')
    # ----------------------------- 1. 导入并转换数据 -----------------------------
    print('========== [PrepareData] >>>> 开始导入 train.tsv 数据....')
    rawDataWithHeader = sc.textFile(Path + u'data/stumbleupon/train-100.tsv')
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace('\"', ''))
    lines = rData.map(lambda x: x.split('\t'))
    print('========== [PrepareData] >>>> 共计：' + str(lines.count()) + ' 项')
    # ----------------------------- 2. 建立训练评估所需数据RDD[LabeledPoint] -----------------------------
    # categoriesMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
    # labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, categoriesMap, -1)))
    print('========== [PrepareData] >>>> 标准化之前：'),
    categoriesMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
    labelRDD = lines.map(lambda r: extract_label(r))
    featureRDD = lines.map(lambda r: extract_features(r, categoriesMap, len(r) - 1))
    for i in featureRDD.first():
        print('\t\t' + str(i) + '(' + str(type(i)) + '),'),
    print('')
    print('========== [PrepareData] >>>> 标准化之后：'),
    stdScaler = StandardScaler(withMean=True, withStd=True).fit(featureRDD)  # 创建标准化刻度
    ScalerFeatureRDD = stdScaler.transform(featureRDD)
    for i in ScalerFeatureRDD.first():
        print('\t\t' + str(i) + '(' + str(type(i)) + '),'),
    labelpoint = labelRDD.zip(ScalerFeatureRDD)  # 使用zip将label与标准化后的特征字段结合起来建立labelpoint
    labelpointRDD = labelpoint.map(lambda r: LabeledPoint(r[0], r[1]))
    # ----------------------------- 3. 以随机方式将数据分为3个部分并返回 -----------------------------
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print('========== [PrepareData] >>>> 将数据以随机方式差分为三个部分：trainData: ' + str(trainData.count()) + ' 项, validationData: ' + str(validationData.count()) + ' 项, testData: ' + str(testData.count()) + ' 项')
    # ----------------------------- 4. 返回元组数据 -----------------------------
    return (trainData, validationData, testData, categoriesMap)

def extract_label(field):
    '''
    提取数据中的label特征字段
    :param field:
    :return:
    '''
    label = (field[-1])
    return float(label)

def extract_features(field, categoriesMap, featureEnd):
    '''
    提取数据的feature特征字段
    :param field:
    :param categoriesMap:
    :param featureEnd:
    :return:
    '''
    # 提取分类特征字段
    categoryIdx = categoriesMap[field[3]]
    categoryFeatures = np.zeros(len(categoriesMap))
    categoryFeatures[categoryIdx] = 1
    # 提取数值字段
    numericalFeatures = [convert_float(field) for field in field[4:featureEnd]]
    # 返回 “分类特征字段” + “数值特征字段”
    return np.concatenate((categoryFeatures, numericalFeatures))

def convert_float(x):
    '''
    将内容转换为float类型数据
    :param x:
    :return:
    '''
    return (0 if x == '?' else float(x))


def PredictData(sc, model, categoriesMap):
    '''
    对网址是暂时性的（ephemeral）或长青的（evergreen）进行预测的函数
    :param sc:
    :param model:
    :param categoriesMap:
    :return:
    '''
    print('======================= 预测数据 =======================')
    print('========== [PredictData] >>>> 开始导入 test.tsv 数据....')
    rawDataWithHeader = sc.textFile(Path + u'data/stumbleupon/test.tsv')
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace('\"', ''))
    lines = rData.map(lambda x: x.split('\t'))
    print('========== [PredictData] >>>> 共计：' + str(lines.count()) + ' 项')
    dataRDD = lines.map(lambda r: (r[0], extract_features(r, categoriesMap, len(r))))
    DescDict = {
        0: '暂时性网页(ephemeral)',
        1: '长青网页(evergreen)'
    }
    for data in dataRDD.take(10):
        predictResult = model.predict(data[1])
        print('========== [PredictData] >>>> 网址：' + str(data[0]) + '\n' +
              '\t\t==>> 预测：' + str(predictResult) + ', 说明：' + DescDict[predictResult] + '\n')

def evaluateModel(model, validationData):
    '''
    使用AUC(Area under of Curve of ROC)评估模型的准确率
    :param model:
    :param validationData:
    :return:
    '''
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC = metrics.areaUnderROC
    return (AUC)


def trainEvaluateModel(trainData, validationData, numIterations, stepSize, regParam):
    '''
    训练模型时会输入不同的参数。其中，DecisionTree参数有impurity、maxDepth、maxBins等的值都会影响准确率以及训练所需的时间。
    我们以图表显示这些参数值、准确率与训练所需的时间。
    我们每次只会评估单个参数的不同值，例如评估maxDepth参数的不同值[3, 5, 10, 15, 20, 25]，执行步骤如下：
    （1）用SVMWithSGD.train进行训练传入trainData与单个参数的不同数值；
    （2）建立模型后，用validationData评估模型的AUC准确率；
    （3）训练与评估模型重复执行多次，产生多个参数项的AUC与运行时间，并存储于metricsRDD中；
    （4）全部执行完成后，将metricsRDD转换为Pandas DataFrame；
    （5）Pandas DataFrame可绘制AUC与运行时间图表，用于显示不同参数的准确率与执行时间的关系。
    :param trainData:
    :param validationData:
    :param numIterations:
    :param stepSize:
    :param regParam:
    :return:
    '''
    print('======================= 训练评估模型 =======================')
    startTime = time()
    model = SVMWithSGD.train(trainData, numIterations, stepSize, regParam)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print('========== [trainEvaluateModel] >>>> 训练评估模型：使用参数：numIterations=' + str(numIterations) + ', stepSize=' + str(stepSize) + ', regParam=' + str(regParam) + '\n' +
          '\t\t==>> 所需时间=' + str(duration) + ', 结果AUC=' + str(AUC))
    return (AUC, duration, numIterations, stepSize, regParam, model)

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
    df = pd.DataFrame(metrics, index=indexList, columns=['AUC', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    return df

def showChart(df, evalParm, barData, lineData, yMin, yMax):
    '''
    用Matplotlib绘制训练评估结果图形
    # 调用示例：showChart(showDataTable(trainData, validationData, ...), 'impurity', 'AUC', 'duration', 0.5, 0.7)
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

def evalParameter(trainData, validationData, evalParm, numIterationsList, stepSizeList, regParamList):
    '''
    评估影响模型准确性的不同参数
    # 调用示例：
    evalParameter(trainData, validationData, 'numIterations',
                    numIterationsList=[1, 3, 5, 10, 15, 20, 25],
                    stepSizeList=[100],
                    regParamList=[1])

    :param trainData:
    :param validationData:
    :param evalParm:
    :param numIterationsList:
    :param stepSizeList:
    :param regParamList:
    :return:
    '''
    print('======================= 评估模型参数(' + evalParm + ') =======================')
    # 训练评估参数
    metrics = [trainEvaluateModel(trainData, validationData, numIteration, stepSize, regParam)
               for numIteration in numIterationsList
               for stepSize in stepSizeList
               for regParam in regParamList]
    # 设置当前评估的参数
    if evalParm == 'numIteration':
        indexList = numIterationsList[:]
    elif evalParm == 'stepSize':
        indexList = stepSizeList[:]
    elif evalParm == 'regParam':
        indexList = regParamList[:]
    # 转换为Pandas DataFrame
    df = pd.DataFrame(metrics, index=indexList, columns=['AUC', 'duration', 'numIteration', 'stepSize', 'regParam', 'model'])
    # 显示图形
    showChart(df, evalParm, 'AUC', 'duration', 0.5, 0.7)

def parameterEval(trainData, validationData):
    '''
    评估'impurity', 'maxDepth', 'maxBins'等参数
    :param trainData:
    :param validationData:
    :return:
    '''
    # 评估 numIterations 参数
    evalParameter(trainData, validationData, 'numIterations',
                  numIterationsList=[1, 3, 5, 10, 15, 20, 25],
                  stepSizeList=[100],
                  regParamList=[1])
    # 评估 stepSize 参数
    evalParameter(trainData, validationData, 'stepSize',
                  numIterationsList=[25],
                  stepSizeList=[10, 50, 100, 200],
                  miniBatchFractionList=[1])
    # 评估 regParam 参数
    evalParameter(trainData, validationData, 'regParam',
                  numIterationsList=[25],
                  stepSizeList=[100],
                  miniBatchFractionList=[0.01, 0.1, 1])

def evalAllParameter(trainData, validationData, numIterationsList, stepSizeList, regParamList):
    '''
    训练评估所有参数，找出最好的参数组合
    # 调用示例：
    bestModel = evalAllParameter(trainData, validationData,
            [1, 3, 5, 10, 15, 20, 25],
            [10, 50, 100, 200],
            [0.01, 0.1, 1])

    :param trainData:
    :param validationData:
    :param numIterationsList:
    :param stepSizeList:
    :param regParamList:
    :return:
    '''
    print('======================= 训练评估所有参数，找出最好的参数组合 =======================')
    # for循环训练评估所有参数组合
    metrics = [trainEvaluateModel(trainData, validationData, numIterations, stepSize, regParam)
               for numIterations in numIterationsList
               for stepSize in stepSizeList
               for regParam in regParamList]
    # 找出AUC最大的参数组合
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter = Smetrics[0]
    # 显示调校后最佳参数组合
    print('========== [evalAllParameter] >>>> 调校后最佳参数组合：numIterations=' + str(bestParameter[2]) + ', stepSize=' + str(bestParameter[3]) + ', regParam=' + str(bestParameter[4]) + '\n' +
          '\t\t==>> 结果AUC=' + str(bestParameter[0]))
    # 返回最佳模型
    return bestParameter[5]


if __name__ == '__main__':
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ Start RunSVMWithSGDBinary @@@@@@@@@@@@@@@@@@@@@@@@@')
    sc = CreateSparkContext()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据准备阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (trainData, validationData, testData, categoriesMap) = PrepareData(sc)
    # 为了提高运行效率暂将数据保存在内存中
    trainData.persist()
    validationData.persist()
    testData.persist()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 训练评估阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (AUC, duration, numIterations, stepSize, regParam, model) = trainEvaluateModel(trainData, validationData, 3, 50, 1)
    if (len(sys.argv) == 2) and (sys.argv[1] == '-e'):
        parameterEval(trainData, validationData)
    elif (len(sys.argv) == 2) and (sys.argv[1] == '-a'):
        print('@@@@@@@@-------- 所有参数训练评估找出最好的参数组合 --------@@@@@@@@')
        model = evalAllParameter(trainData, validationData,
                                [1, 3, 5, 10, 15, 20, 25],
                                [10, 50, 100, 200],
                                [0.01, 0.1, 1])
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 测试阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    auc = evaluateModel(model, testData)
    print('@@@@@@@@============== [Main] >>>> 使用 testData 测试最佳模型，结果AUC=' + str(auc))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 预测数据 @@@@@@@@@@@@@@@@@@@@@@@@@')
    PredictData(sc, model, categoriesMap)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 查看分类规则 @@@@@@@@@@@@@@@@@@@@@@@@@')
    # print('@@@@@@@@============== [Main] >>>> Print Debug Info: \n')
    print(model.toDebugString())

