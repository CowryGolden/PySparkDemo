#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: MLPipelineRegression.py\n
@time: 2019-10-16 17:07\n
@desc: Spark ML Pipeline 机器学习流程回归分析【TrainValidationSplit、CrossValidation】，以及梯度提升决策树回归器(GBTRegressor)的使用
'''
import  os, sys
# import pandas as pd
# import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor  # Gradient-Boosted Trees Regressor 梯度提升决策树回归器
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator


os.environ['SPARK_HOME'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6'
os.environ['JAVA_HOME'] = 'D:\ProgramFiles\Java\jdk1.8.0_131'
os.environ['HADOOP_HOME'] = 'D:\ProgramFiles\hadoop\hadoop-2.6.4'
os.environ['PYSPARK_PYTHON'] = 'D:\ProgramFiles\Python364\python.exe'  # 在dos命令行下执行时需要，在IDE中运行配置到Run Configurations中即可，只能使用Python364中的python.exe，而Anaconda3中的python.exe报错：Cannot run program "?D:\ProgramFiles\Anaconda3\python.exe": CreateProcess error=2,

def CreateSparkContext():
    '''
    创建SparkContext对象
    :return:
    '''
    sparkConf = SparkConf().setAppName('MLPipelineRegression').set('spark.ui.showConsoleProgress', 'false')
    sc = SparkContext(conf=sparkConf)
    print('master=' + sc.master)
    SetLogger(sc)
    SetPath(sc)
    return sc

def CreateSqlContext():
    '''
    创建SqlContext对象
    :return:
    '''
    sqlContext = SparkSession.builder.getOrCreate()
    return sqlContext

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

def PrepareData(sc, sqlContext):
    '''
    准备数据
    :param sc:
    :param sqlContext:
    :return:
    '''
    print('======================= 准备数据 =======================')
    # ----------------------------- 1. 导入并转换数据 -----------------------------
    print('========== [PrepareData] >>>> 开始导入 hour.csv 数据....')
    # rawData = sc.textFile(Path + u'data/bikesharing/hour.csv')
    # lines = rawData.map(lambda x: x.split(','))
    rowDF = sqlContext.read.format('csv').option('header', 'true').load(Path + u'data/bikesharing/hour.csv')
    # print('========== [PrepareData] >>>> 查看数据 train.tsv DataFrame 的 Schema ....')
    # rowDF.printSchema()
    print('========== [PrepareData] >>>> hour.csv 数据共 ' + str(rowDF.count()) + ' 项。')
    df0 = rowDF.drop('instant').drop('dteday').drop('yr').drop('casual').drop('registered')  # 删除不需要字段
    df = df0.select([col(column).cast('double').alias(column) for column in df0.columns])
    featuresCols = df.columns[:-1]  # 最后一个字段是label，其余都是特征字段
    (train_df, test_df) = df.randomSplit([0.7, 0.3])  # 将数据按照 7:3 的比例分成train_df（训练数据）和test_df（测试数据）
    # print('========== [PrepareData] >>>> 转换后的 DataFrame 数据共 ' + str(df.count()) + ' 项，将其按[7:3]比例分割后，其中train_df（训练数据）共 ' + str(train_df.count()) + ' 项，test_df（测试数据）共 ' + str(test_df.count()) + ' 项。')
    train_df.cache()
    test_df.cache()
    return (df, train_df, test_df, featuresCols)

def getPrePipelineStages(featuresCols):
    '''
    获取 ML Pipeline 流程stages的前几步：
    1、VectorAssembler 可以将多个特征字段整合成一个特征的 Vector
    2、VectorIndexer 可以将一个特征字段生成多个特征字段
    :param featuresCols:
    :return:
    '''
    vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol='aFeatures')
    vectorIndexer = VectorIndexer(inputCol='aFeatures', outputCol='features', maxCategories=24)
    stages = [vectorAssembler, vectorIndexer]
    return stages

def trainAndEvalModelByDecisionTreeRegressor(stages, train_df, test_df, evaluator):
    '''
    使用 DecisionTreeRegressor 决策树回归建立机器学习Pipeline流程进行模型训练和评估
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    print('======================= 使用 DecisionTreeRegressor 建立 ML Pipeline 流程进行模型训练 =======================')
    dt = DecisionTreeRegressor(labelCol='cnt', featuresCol='features')
    dtPipeline = Pipeline(stages=stages+[dt])  # print(str(dtPipeline.getStages()))
    dtPipelineModel = dtPipeline.fit(train_df)
    bestModel = dtPipelineModel.stages[2]  # print(bestModel.toDebugString)
    print('======================= 使用 DecisionTreeRegressor 建立 ML Pipeline 流程进行模型训练后，使用模型进行预测 =======================')
    predicts = dtPipelineModel.transform(test_df)
    # print(str(predicts.columns))  # 预测后新增的字段：'aFeatures', 'features', 'prediction'
    predicts.select('season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt', 'prediction').show(10)
    rmse = evaluator.evaluate(predicts)
    print('======================= 使用 DecisionTreeRegressor 建立 ML Pipeline 流程进行模型训练后，评估模型准确率（rmse=' + str(rmse) + '） =======================')
    return (bestModel, predicts, rmse)

def trainAndEvalModelByDecisionTreeRegressorAndTrainValidationSplit(stages, train_df, test_df, evaluator):
    '''
    使用 DecisionTreeRegressor 决策树回归和 TrainValidationSplit 建立机器学习Pipeline流程进行模型训练和验证，并找出最佳模型
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    print('======================= 使用 DecisionTreeRegressor、TrainValidationSplit 建立 ML Pipeline 流程进行模型训练 =======================')
    dt = DecisionTreeRegressor(labelCol='cnt', featuresCol='features')
    paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15, 25]).addGrid(dt.maxBins, [25, 35, 45, 50]).build()  # 执行模型参数训练 4*4=16次，其中impurity="variance"固定不变，不用再参与训练，由于在line：108，创建 vectorIndexer 时，设置了maxCategories=24，因此这里maxBins要大于24
    tsv = TrainValidationSplit(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid, trainRatio=0.8)
    tsvPipeline = Pipeline(stages=stages+[tsv])
    tsvPipelineModel = tsvPipeline.fit(train_df)
    bestModel = tsvPipelineModel.stages[2].bestModel
    print('======================= 使用 DecisionTreeRegressor、TrainValidationSplit 建立 ML Pipeline 流程进行模型训练后，使用模型进行预测 =======================')
    predicts = tsvPipelineModel.transform(test_df)
    rmse = evaluator.evaluate(predicts)
    print('======================= 使用 DecisionTreeRegressor、TrainValidationSplit 建立 ML Pipeline 流程进行模型训练后，评估模型准确率（rmse=' + str(rmse) + '） =======================')
    return (bestModel, predicts, rmse)

def trainAndEvalModelByDecisionTreeRegressorAndCrossValidator(stages, train_df, test_df, evaluator):
    '''
    使用 DecisionTreeRegressor 决策树回归和 CrossValidator 建立机器学习Pipeline流程进行模型训练和验证，并找出最佳模型
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    print('======================= 使用 DecisionTreeRegressor、CrossValidator 建立 ML Pipeline 流程进行模型训练 =======================')
    dt = DecisionTreeRegressor(labelCol='cnt', featuresCol='features')
    paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15, 25]).addGrid(dt.maxBins, [25, 35, 45, 50]).build()  # 执行模型参数训练 4*4=16次，其中impurity="variance"固定不变，不用再参与训练，由于在line：108，创建 vectorIndexer 时，设置了maxCategories=24，因此这里maxBins要大于24
    cv = CrossValidator(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)
    cvPipeline = Pipeline(stages=stages+[cv])
    cvPipelineModel = cvPipeline.fit(train_df)
    bestModel = cvPipelineModel.stages[2].bestModel
    print('======================= 使用 DecisionTreeRegressor、CrossValidator 建立 ML Pipeline 流程进行模型训练后，使用模型进行预测 =======================')
    predicts = cvPipelineModel.transform(test_df)
    rmse = evaluator.evaluate(predicts)
    print('======================= 使用 DecisionTreeRegressor、CrossValidator 建立 ML Pipeline 流程进行模型训练后，评估模型准确率（rmse=' + str(rmse) + '） =======================')
    return (bestModel, predicts, rmse)

def trainAndEvalModelByGBTRegressorAndCrossValidator(stages, train_df, test_df, evaluator):
    '''
    使用 GBTRegressor 梯度提升决策树回归和 CrossValidator 建立机器学习Pipeline流程进行模型训练和验证，并找出最佳模型
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    print('======================= 使用 GBTRegressor、CrossValidator 建立 ML Pipeline 流程进行模型训练 =======================')
    gbt = GBTRegressor(labelCol='cnt', featuresCol='features')
    paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10]).addGrid(gbt.maxBins, [25, 40]).addGrid(gbt.maxIter, [10, 50]).build()  # 执行模型参数训练 4*4*2=32次，其中impurity="variance"固定不变，不用再参与训练，由于在line：108，创建 vectorIndexer 时，设置了maxCategories=24，因此这里maxBins要大于24
    cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)
    cvPipeline = Pipeline(stages=stages+[cv])
    cvPipelineModel = cvPipeline.fit(train_df)
    bestModel = cvPipelineModel.stages[2].bestModel
    print('======================= 使用 GBTRegressor、CrossValidator 建立 ML Pipeline 流程进行模型训练后，使用模型进行预测 =======================')
    predicts = cvPipelineModel.transform(test_df)
    predicts.select('season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt', 'prediction').show(10)
    rmse = evaluator.evaluate(predicts)
    print('======================= 使用 GBTRegressor、CrossValidator 建立 ML Pipeline 流程进行模型训练后，评估模型准确率（rmse=' + str(rmse) + '） =======================')
    return (bestModel, predicts, rmse)


if __name__ == '__main__':
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ Start MLPipelineRegression @@@@@@@@@@@@@@@@@@@@@@@@@')
    sc = CreateSparkContext()
    sqlContext = CreateSqlContext()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据准备阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (df, train_df, test_df, featuresCols) = PrepareData(sc, sqlContext)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据训练阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    stages = getPrePipelineStages(featuresCols)
    evaluator = RegressionEvaluator(labelCol='cnt', predictionCol='prediction', metricName='rmse')  # 创建决策树回归模型评估器
    (bestModel, predicts, rmse) = trainAndEvalModelByDecisionTreeRegressor(stages, train_df, test_df, evaluator)  # 使用决策树回归建立机器学习Pipeline流程进行模型训练和评估
    print()
    (bestModel, predicts, rmse) = trainAndEvalModelByDecisionTreeRegressorAndTrainValidationSplit(stages, train_df, test_df, evaluator)  # 使用 DecisionTreeRegressor 决策树回归和 TrainValidationSplit 建立机器学习Pipeline流程进行模型训练和验证，并找出最佳模型
    print()
    (bestModel, predicts, rmse) = trainAndEvalModelByDecisionTreeRegressorAndCrossValidator(stages, train_df, test_df, evaluator) # 使用 DecisionTreeRegressor 决策树回归和 CrossValidator 建立机器学习Pipeline流程进行模型训练和验证，并找出最佳模型
    print()
    (bestModel, predicts, rmse) = trainAndEvalModelByGBTRegressorAndCrossValidator(stages, train_df, test_df, evaluator)  # 使用 GBTRegressor 梯度提升决策树回归和 CrossValidator 建立机器学习Pipeline流程进行模型训练和验证，并找出最佳模型
