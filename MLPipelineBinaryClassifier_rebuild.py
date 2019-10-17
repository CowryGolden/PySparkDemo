#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: MLPipelineBinaryClassifier.py\n
@time: 2019-10-15 11:44\n
@desc: Spark ML Pipeline 机器学习流程二元分类【TrainValidationSplit、CrossValidation】，以及使用随机森林分类器（RandomForestClassifier）——代码重构
'''
import  os, sys
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf, col
import pyspark.sql.types
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer  # StringIndexer 可以将文字的分类特征字段转换为数字，功能类似于categoriesMap
from pyspark.ml.feature import OneHotEncoder  # OneHotEncoder 可以将一个数值的分类特征字段转换为多个字段的Vector
from pyspark.ml.feature import VectorAssembler  # VectorAssembler 可以将多个特征字段整合成一个特征的 Vector
from pyspark.ml.classification import DecisionTreeClassifier  # 可以进行二元分类
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # 用于评估模型的准确率
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit  # TrainValidation 模块可以进行模型训练验证，并找出最佳模型
from pyspark.ml.tuning import CrossValidator  # 用于交叉验证，找出最佳模型
from pyspark.ml.classification import RandomForestClassifier  # 可以进行随机森林分类


os.environ['SPARK_HOME'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6'
os.environ['JAVA_HOME'] = 'D:\ProgramFiles\Java\jdk1.8.0_131'
os.environ['HADOOP_HOME'] = 'D:\ProgramFiles\hadoop\hadoop-2.6.4'
os.environ['PYSPARK_PYTHON'] = 'D:\ProgramFiles\Python364\python.exe'  # 在dos命令行下执行时需要，在IDE中运行配置到Run Configurations中即可，只能使用Python364中的python.exe，而Anaconda3中的python.exe报错：Cannot run program "?D:\ProgramFiles\Anaconda3\python.exe": CreateProcess error=2,

def CreateSparkContext():
    '''
    创建SparkContext对象
    :return:
    '''
    sparkConf = SparkConf().setAppName('MLPipelineBinaryClassifier').set('spark.ui.showConsoleProgress', 'false')
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
    print('========== [PrepareData] >>>> 开始导入 train.tsv 数据....')
    # rawDataWithHeader = sc.textFile(Path + u'data/stumbleupon/train.tsv')
    rowDF = sqlContext.read.format('csv').option('header', 'true').option('delimiter', '\t').load(Path + u'data/stumbleupon/train.tsv')
    print('========== [PrepareData] >>>> train.tsv 数据共 ' + str(rowDF.count()) + ' 项')
    # print('========== [PrepareData] >>>> 查看数据 train.tsv DataFrame 的 Schema ....')
    # rowDF.printSchema()
    # rowDF.select('url', 'alchemy_category', 'alchemy_category_score', 'is_news', 'label').show(10)
    return (rowDF)

def replace_question(x):
    '''
    将问号转换为字符串'0'
    :param x:
    :return:
    '''
    return ('0' if x == '?' else x)

def createPreTrainStages(rowDF, df):
    '''
    创建训练模型stages的前三步
    1、StringIndexer 可以将文字的分类特征字段转换为数字，功能类似于categoriesMap
    2、OneHotEncoder 可以将一个数值的分类特征字段转换为多个字段的Vector
    3、VectorAssembler 可以将多个特征字段整合成一个特征的 Vector
    :param rowDF:
    :param df:
    :return:
    '''
    categoryIndexer = StringIndexer(inputCol='alchemy_category', outputCol='alchemy_category_index')  # StringIndexer 可以将文字的分类特征字段转换为数字，功能类似于categoriesMap
    categoryTransformer = categoryIndexer.fit(df)
    # 查看网页分类字典（或对照表）——categoryTransformer.labels
    # print('========== [Main] >>>> 查看网页分类字典（或对照表）——categoryTransformer.labels')
    # for i in range(0, len(categoryTransformer.labels)):
    #     print(str(i) + ':' + categoryTransformer.labels[i])

    df1 = categoryTransformer.transform(train_df)
    # print(df1.columns)  # 发现新增了'alchemy_category_index'字段
    # df1.select('alchemy_category', 'alchemy_category_index').show(10)

    encoder = OneHotEncoder(dropLast=False, inputCol='alchemy_category_index', outputCol='alchemy_category_indexVec')  # OneHotEncoder 可以将一个数值的分类特征字段转换为多个字段的Vector
    df2 = encoder.transform(df1)
    # print(df2.columns)  # 发现新增了'alchemy_category_indexVec'字段
    # df2.select('alchemy_category', 'alchemy_category_index', 'alchemy_category_indexVec').show(10)

    assemblerInputs = ['alchemy_category_indexVec'] + rowDF.columns[4:-1]  # 创建全部特征字段的assemblerInputs：之前生成的分类特征字段 alchemy_category_index(14个字段的数值)，加上原本第4个字段到倒数第2个字段的数值特征字段
    # print(assemblerInputs)
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol='features')  # VectorAssembler 可以将多个特征字段整合成一个特征的 Vector
    df3 = assembler.transform(df2)
    # print(df3.columns)  # 整合后新增了字段'features'
    # df3.select('url', 'alchemy_category', 'alchemy_category_index', 'alchemy_category_indexVec', 'features').show(10)
    # print(str(df3.select('features').take(1)))
    stages = [categoryIndexer, encoder, assembler]
    return (df3, stages)

def trainAndEvalModelByDecisionTreeClassifier(stages, train_df, test_df, evaluator):
    '''
    使用 DecisionTreeClassifier 决策树分类器训练评估模型
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    dt = DecisionTreeClassifier(labelCol='label', featuresCol='features', impurity='gini', maxDepth=10, maxBins=14)  # 已经准备好了具有label和features字段的数据，使用DecisionTreeClassifier进行二元分类
    # dtModel = dt.fit(df3)
    # print(str(dtModel))  # DecisionTreeClassificationModel (uid=DecisionTreeClassifier_d60cee32e188) of depth 10 with 537 nodes
    # df4 = dtModel.transform(df3)

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@ Pipeline建立阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    pipeline = Pipeline(stages=stages + [dt])  # 建立pipeline；其中建立pipeline不需要花太多时间，但后续pipeline.fit()进行训练时会花很多时间
    # print('========== [Main] >>>> 查看机器学习 pipeline 流程的每一个阶段：' + str(pipeline.getStages()))  # [StringIndexer_23f610ef8e7c, OneHotEncoder_28258e4bbc11, VectorAssembler_9fd7bf7e10f2, DecisionTreeClassifier_8b02af727e7a]

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据训练阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    pipelineModel = pipeline.fit(train_df)
    # print('========== [Main] >>>> 查看训练完成后的决策树模型：' + str(pipelineModel.stages[3]))  # 训练完成后产生的模型为：pipelineModel，训练过程的第3阶段会产生决策树模型
    # print('========== [Main] >>>> 查看训练完成后的决策树模型规则：')
    # print(pipelineModel.stages[3].toDebugString)
    bestModel = pipelineModel.stages[3]

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据预测阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    # predicted = pipelineModel.transform(test_df)
    # print(str(predicted.columns))  # 预测后新增的字段：'rawPrediction', 'probability', 'prediction'
    # predicted.select('url', 'features', 'rawPrediction', 'probability', 'label', 'prediction').show(10)  # 查看预测结果
    # print(str(predicted.select('probability', 'prediction').take(10)))

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@ 使用二元分类评估器验证评估模型 @@@@@@@@@@@@@@@@@@@@@@@@@')
    predictions = pipelineModel.transform(test_df)  # 使用训练好的模型传入测试数据test_df进行预测，产生预测结果predictions
    auc = evaluator.evaluate(predictions)  # 计算AUC
    return (bestModel, predictions, auc)

def trainAndEvalModelByDecisionTreeClassifierAndTrainValidationSplit(stages, train_df, test_df, evaluator):
    '''
    使用 DecisionTreeClassifier 决策树分类器和 TrainValidationSplit 进行模型训练和验证，并找出最佳模型
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    dt = DecisionTreeClassifier(labelCol='label', featuresCol='features', impurity='gini', maxDepth=10, maxBins=14)
    paramGrid = ParamGridBuilder().addGrid(dt.impurity, ['gini', 'entropy']).addGrid(dt.maxDepth, [5, 10, 15]).addGrid(dt.maxBins, [10, 15, 20]).build()  # 执行模型参数训练 2*3*3=18次
    tvs = TrainValidationSplit(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid, trainRatio=0.8)  # 创建模型训练验证对象；参数 trainRatio=0.8 表示：训练验证前会将数据按照 8:2 的比例分成训练数据与验证数据
    tvsPipline = Pipeline(stages=stages+[tvs])  # 建立模型训练 Pipeline 流程
    tvsPiplineModel = tvsPipline.fit(train_df)  # 生成训练后的模型
    bestModel = tvsPiplineModel.stages[3].bestModel
    # print('========== [trainAndEvalModelByTrainValidationSplit] >>>> 查看训练完成后的最佳决策树模型规则：')
    # print(bestModel.toDebugString[:500])  # 只显示前500个字符
    predictions = tvsPiplineModel.transform(test_df)
    auc = evaluator.evaluate(predictions)
    return (bestModel, predictions, auc)

def trainAndEvalModelByDecisionTreeClassifierAndCrossValidator(stages, train_df, test_df, evaluator):
    '''
    使用 DecisionTreeClassifier 决策树分类器和 CrossValidator 交叉验证训练模型，并找出最佳模型阶段
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    dt = DecisionTreeClassifier(labelCol='label', featuresCol='features', impurity='gini', maxDepth=10, maxBins=14)
    paramGrid = ParamGridBuilder().addGrid(dt.impurity, ['gini', 'entropy']).addGrid(dt.maxDepth, [5, 10, 15]).addGrid(dt.maxBins, [10, 15, 20]).build()  # 执行模型参数训练 2*3*3=18次
    cv = CrossValidator(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)
    cvPipline = Pipeline(stages=stages+[cv])
    cvPiplineModel = cvPipline.fit(train_df)
    bestModel = cvPiplineModel.stages[3].bestModel
    # print('========== [Main] >>>> 查看训练完成后的最佳决策树模型规则：')
    # print(bestModel1.toDebugString[:500])  # 只显示前500个字符
    predictions = cvPiplineModel.transform(test_df)
    auc = evaluator.evaluate(predictions)
    return (bestModel, predictions, auc)

def trainAndEvalModelByRandomForestClassifier(stages, train_df, test_df, evaluator):
    '''
    使用 RandomForestClassifier 随机森林分类器训练评估模型
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=10)
    rfPipeline = Pipeline(stages=stages+[rf])
    rfPipelineModel = rfPipeline.fit(train_df)
    bestModel = rfPipelineModel.stages[3]
    predictions = rfPipelineModel.transform(test_df)
    auc = evaluator.evaluate(predictions)
    return (bestModel, predictions, auc)

def trainAndEvalModelByRandomForestClassifierAndTrainValidationSplit(stages, train_df, test_df, evaluator):
    '''
    使用 RandomForestClassifier 分类器和 TrainValidationSplit 训练验证模型，并找出最佳模型
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=10)
    paramGrid = ParamGridBuilder().addGrid(rf.impurity, ['gini', 'entropy']).addGrid(rf.maxDepth, [5, 10, 15]).addGrid(rf.maxBins, [10, 15, 20]).addGrid(rf.numTrees, [10, 20, 30]).build()
    rftvs = TrainValidationSplit(estimator=rf, evaluator=evaluator, estimatorParamMaps=paramGrid, trainRatio=0.8)
    rftvsPipeline = Pipeline(stages=stages+[rftvs])
    rftvsPipelineModel = rftvsPipeline.fit(train_df)
    bestModel = rftvsPipelineModel.stages[3].bestModel
    predictions = rftvsPipelineModel.transform(test_df)
    auc = evaluator.evaluate(predictions)
    return (bestModel, predictions, auc)

def trainAndEvalModelByRandomForestClassifierAndCrossValidator(stages, train_df, test_df, evaluator):
    '''
    使用 RandomForestClassifier 分类器和 TrainValidationSplit 训练验证模型，并找出最佳模型
    :param stages:
    :param train_df:
    :param test_df:
    :param evaluator:
    :return:
    '''
    rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=10)
    paramGrid = ParamGridBuilder().addGrid(rf.impurity, ['gini', 'entropy']).addGrid(rf.maxDepth, [5, 10, 15]).addGrid(rf.maxBins, [10, 15, 20]).addGrid(rf.numTrees, [10, 20, 30]).build()
    rfcv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)
    rfcvPipeline = Pipeline(stages=stages+[rfcv])
    rftvsPipelineModel = rfcvPipeline.fit(train_df)
    bestModel = rftvsPipelineModel.stages[3].bestModel
    predictions = rftvsPipelineModel.transform(test_df)
    auc = evaluator.evaluate(predictions)
    return (bestModel, predictions, auc)


if __name__ == '__main__':
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ Start MLPipelineBinaryClassifier @@@@@@@@@@@@@@@@@@@@@@@@@')
    sc = CreateSparkContext()
    sqlContext = CreateSqlContext()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据准备阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (rowDF) = PrepareData(sc, sqlContext)
    replace_question = udf(replace_question)  # 使用 udf 将 replace_question 转换为 DataFrame UDF 用户自定义函数
    df = rowDF.select(['url', 'alchemy_category'] + [replace_question(col(column)).cast('double').alias(column) for column in rowDF.columns[4:]])  # 把rowDF第4个字段至最后一个字段转换为double类型，最后一个字段是label，其余是feature
    # print('========== [Main] >>>> 查看 DataFrame 使用 replace_question UDF 转换后的 Schema ....')
    # df.printSchema()
    (train_df, test_df) = df.randomSplit([0.7, 0.3])  # 将 df 按照 7:3 的比例分成 train_df（训练数据） 与 test_df（测试数据），并暂存在内存中
    train_df.cache()
    test_df.cache()

    (df3, stages) = createPreTrainStages(rowDF, df)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC')  # 用于评估模型的准确率，使用AUC方式判断

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@ 使用 DecisionTreeClassifier 决策树分类器训练评估模型 @@@@@@@@@@@@@@@@@@@@@@@@@')
    # (bestModel, predictions, auc) = trainAndEvalModelByDecisionTreeClassifier(stages, train_df, test_df, evaluator)
    # print('========== [Main] >>>> 执行完使用 DecisionTreeClassifier 决策树分类器训练评估模型后，使用 testData 评估预测模型，结果AUC=' + str(auc))

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@ 使用 DecisionTreeClassifier 决策树分类器和 TrainValidationSplit 进行模型训练和验证，并找出最佳模型阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    # (bestModel, predictions, auc) = trainAndEvalModelByDecisionTreeClassifierAndTrainValidationSplit(stages, train_df, test_df, evaluator)
    # print('========== [Main] >>>> 执行完使用 DecisionTreeClassifier 决策树分类器和 TrainValidationSplit 进行模型训练和验证后，使用 testData 评估预测模型，结果AUC=' + str(auc))

    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 使用 DecisionTreeClassifier 决策树分类器和 CrossValidator 交叉验证训练模型，并找出最佳模型阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (bestModel, predictions, auc) = trainAndEvalModelByDecisionTreeClassifierAndCrossValidator(stages, train_df,test_df, evaluator)
    print('========== [Main] >>>> 执行完使用 DecisionTreeClassifier 决策树分类器和 CrossValidator 交叉验证训练模型后，使用 testData 评估预测模型，结果AUC=' + str(auc))

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@ 使用 RandomForestClassifier 随机森林分类器训练评估模型阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    # (bestModel, predictions, auc) = trainAndEvalModelByRandomForestClassifier(stages, train_df, test_df, evaluator)
    # print('========== [Main] >>>> 执行完使用 RandomForestClassifier 随机森林分类器训练评估模型后，使用 testData 评估预测模型，结果AUC=' + str(auc))

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@ 使用 RandomForestClassifier 分类器和 TrainValidationSplit 训练验证模型，并找出最佳模型阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    # (bestModel, predictions, auc) = trainAndEvalModelByRandomForestClassifierAndTrainValidationSplit(stages, train_df, test_df, evaluator)
    # print('========== [Main] >>>> 执行完使用 RandomForestClassifier 分类器和 TrainValidationSplit 训练验证模型后，使用 testData 评估预测模型，结果AUC=' + str(auc))

    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 使用 RandomForestClassifier 分类器和 CrossValidator 训练验证模型，并找出最佳模型阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (bestModel, predictions, auc) = trainAndEvalModelByRandomForestClassifierAndCrossValidator(stages, train_df, test_df, evaluator)
    print('========== [Main] >>>> 执行完使用 RandomForestClassifier 分类器和 CrossValidator 训练验证模型后，使用 testData 评估预测模型，结果AUC=' + str(auc))
    DescDict = {
        0: '暂时性网页(ephemeral)',
        1: '长青网页(evergreen)'
    }
    for data in predictions.select('url', 'prediction').take(10):
        print('========== [Main] >>>> 网址：' + str(data[0]) + '\n' +
              '\t\t==>> 预测：' + str(data[1]) + ', 说明：' + DescDict[data[1]] + '\n')


