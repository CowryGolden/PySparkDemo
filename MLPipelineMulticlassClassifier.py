#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: MLPipelineMulticlassClassifier.py.py\n
@time: 2019-10-16 14:59\n
@desc: Spark ML Pipeline 机器学习流程多元分类【TrainValidationSplit】
'''
import  os, sys
# import pandas as pd
# import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

os.environ['SPARK_HOME'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6'
os.environ['JAVA_HOME'] = 'D:\ProgramFiles\Java\jdk1.8.0_131'
os.environ['HADOOP_HOME'] = 'D:\ProgramFiles\hadoop\hadoop-2.6.4'
os.environ['PYSPARK_PYTHON'] = 'D:\ProgramFiles\Python364\python.exe'  # 在dos命令行下执行时需要，在IDE中运行配置到Run Configurations中即可，只能使用Python364中的python.exe，而Anaconda3中的python.exe报错：Cannot run program "?D:\ProgramFiles\Anaconda3\python.exe": CreateProcess error=2,

def CreateSparkContext():
    '''
    创建SparkContext对象
    :return:
    '''
    sparkConf = SparkConf().setAppName('MLPipelineMulticlassClassifier').set('spark.ui.showConsoleProgress', 'false')
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
    print('========== [PrepareData] >>>> 开始导入 covtype.data 数据....')
    rawData = sc.textFile(Path + u'data/covertype/covtype.data')
    # rowDF = sqlContext.read.format('csv').option('header', 'false').option('delimiter', ',').load(Path + u'data/covertype/covtype.data')
    # print('========== [PrepareData] >>>> 查看数据 covtype.data DataFrame 的 Schema ....')
    # rowDF.printSchema()
    lines = rawData.map(lambda x: x.split(','))
    print('========== [PrepareData] >>>> covtype.data 数据共 ' + str(lines.count()) + ' 项。')
    fieldCounts = len(lines.first())  # 计算字段个数
    fields = [StructField('f' + str(i) , StringType(), True) for i in range(fieldCounts)]
    schema = StructType(fields)  # 创建Fields的schema
    df0 = sqlContext.createDataFrame(lines, schema)
    # print(str(df0.columns))  # 55个字段   # df0.printSchema()
    df1 = df0.select([col(column).cast('double').alias(column) for column in df0.columns])  # df1.printSchema()
    featuresCols = df1.columns[:54]  # 前53个字段为特征feature，第54个字段为label
    df = df1.withColumn('label', df1['f54'] - 1).drop('f54')  # 创建label字段并且其值减1，label值范围0~6  # df.show(1)  # 查看第一项数据
    (train_df, test_df) = df.randomSplit([0.7, 0.3])  # 将数据按照 7:3 的比例分成train_df（训练数据）和test_df（测试数据）
    # print('========== [PrepareData] >>>> 转换后的 DataFrame 数据共 ' + str(df.count()) + ' 项，将其按[7:3]比例分割后，其中train_df（训练数据）共 ' + str(train_df.count()) + ' 项，test_df（测试数据）共 ' + str(test_df.count()) + ' 项。')
    train_df.cache()
    test_df.cache()
    return (df, train_df, test_df, featuresCols)

def getPrePipelineStages(featuresCols):
    '''
    获取 ML Pipeline 流程stages的前几步：
    1、VectorAssembler 可以将多个特征字段整合成一个特征的 Vector
    :param featuresCols:
    :return:
    '''
    vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol='features')
    stages = [vectorAssembler]
    return stages

def trainAndEvalModelByDecisionTreeClassifier(stages, train_df, test_df, evaluator):
    '''
    使用 DecisionTreeClassifier 决策树分类建立机器学习Pipeline流程进行模型训练和评估
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    print('======================= 使用 DecisionTreeClassifier 建立 ML Pipeline 流程进行模型训练 =======================')
    dt = DecisionTreeClassifier(labelCol='label', featuresCol='features', maxDepth=5, maxBins=20)
    dtPipeline = Pipeline(stages=stages+[dt])  # print(str(dtPipeline.getStages()))
    dtPipelineModel = dtPipeline.fit(train_df)
    bestModel = dtPipelineModel.stages[1]  # print(bestModel.toDebugString)
    print('======================= 使用 DecisionTreeClassifier 建立 ML Pipeline 流程进行模型训练后，使用模型进行预测 =======================')
    predicts = dtPipelineModel.transform(test_df)
    # print(str(predicts.columns))  # 预测后新增的字段：'rawPrediction', 'probability', 'prediction'
    # predicts.select('probability', 'prediction').show(10)
    accuracy = evaluator.evaluate(predicts)
    print('======================= 使用 DecisionTreeClassifier 建立 ML Pipeline 流程进行模型训练后，评估模型准确率（accuracy=' + str(accuracy) + '） =======================')
    return (bestModel, predicts, accuracy)

def trainAndEvalModelByDecisionTreeClassifierAndTrainValidationSplit(stages, train_df, test_df, evaluator):
    '''
    使用 DecisionTreeClassifier 决策树分类和 TrainValidationSplit 建立机器学习Pipeline流程进行模型训练和验证，并找出最佳模型
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    print('======================= 使用 DecisionTreeClassifier、TrainValidationSplit 建立 ML Pipeline 流程进行模型训练 =======================')
    dt = DecisionTreeClassifier(labelCol='label', featuresCol='features', maxDepth=5, maxBins=20)
    paramGrid = ParamGridBuilder().addGrid(dt.impurity, ['gini', 'entropy']).addGrid(dt.maxDepth, [10, 15, 25]).addGrid(dt.maxBins, [30, 40, 50]).build()  # 执行模型参数训练 2*3*3=18次
    tsv = TrainValidationSplit(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid, trainRatio=0.8)
    tsvPipeline = Pipeline(stages=stages+[tsv])
    tsvPipelineModel = tsvPipeline.fit(train_df)
    bestModel = tsvPipelineModel.stages[1].bestModel
    print('======================= 使用 DecisionTreeClassifier、TrainValidationSplit 建立 ML Pipeline 流程进行模型训练后，使用模型进行预测 =======================')
    predicts = tsvPipelineModel.transform(test_df)
    accuracy = evaluator.evaluate(predicts)
    print('======================= 使用 DecisionTreeClassifier、TrainValidationSplit 建立 ML Pipeline 流程进行模型训练后，评估模型准确率（accuracy=' + str(accuracy) + '） =======================')
    return (bestModel, predicts, accuracy)




if __name__ == '__main__':
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ Start MLPipelineMulticlassClassifier @@@@@@@@@@@@@@@@@@@@@@@@@')
    sc = CreateSparkContext()
    sqlContext = CreateSqlContext()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据准备阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (df, train_df, test_df, featuresCols) = PrepareData(sc, sqlContext)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据训练阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    stages = getPrePipelineStages(featuresCols)
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')  # 创建多元分类模型评估器
    (bestModel, predicts, accuracy) = trainAndEvalModelByDecisionTreeClassifier(stages, train_df, test_df, evaluator)  # 使用决策树分类建立机器学习Pipeline流程进行模型训练和评估
    print()
    (bestModel, predicts, accuracy) = trainAndEvalModelByDecisionTreeClassifierAndTrainValidationSplit(stages, train_df, test_df, evaluator)  # 使用 DecisionTreeClassifier 决策树分类和 TrainValidationSplit 建立机器学习Pipeline流程进行模型训练和验证，并找出最佳模型
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 使用最佳模型预测阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    result = predicts.withColumnRenamed('f0', '海拔') \
                    .withColumnRenamed('f1', '方位') \
                    .withColumnRenamed('f2', '斜率') \
                    .withColumnRenamed('f3', '垂直距离') \
                    .withColumnRenamed('f4', '水平距离') \
                    .withColumnRenamed('f5', '9点时阴影')
    result.select('海拔', '方位', '斜率', '垂直距离', '水平距离', '9点时阴影', 'label', 'prediction').show(10)

