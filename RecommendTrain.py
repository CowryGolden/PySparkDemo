#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: RecommendTrain.py\n
@time: 2019-09-30 16:48\n
@desc: 使用pyspark创建推荐引擎，训练电影推荐数据并存储模型
'''
import  os, sys
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import Rating, ALS

os.environ['SPARK_HOME'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6'
os.environ['JAVA_HOME'] = 'D:\ProgramFiles\Java\jdk1.8.0_131'
os.environ['HADOOP_HOME'] = 'D:\ProgramFiles\hadoop\hadoop-2.6.4'
os.environ['PYSPARK_PYTHON'] = 'D:\ProgramFiles\Python364\python.exe'  # 在dos命令行下执行时需要，在IDE中运行配置到Run Configurations中即可，只能使用Python364中的python.exe，而Anaconda3中的python.exe报错：Cannot run program "?D:\ProgramFiles\Anaconda3\python.exe": CreateProcess error=2,

def CreateSparkContext():
    '''
    创建SparkContext对象
    :return:
    '''
    sparkConf = SparkConf().setAppName('RecommendTrain').set('spark.ui.showConsoleProgress', 'false')
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
    :return:
    '''
    rawUserData = sc.textFile(Path + u'data/ml-100k/u.data')
    rawRatings = rawUserData.map(lambda line: line.split('\t')[:3])
    ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))
    return ratingsRDD


def SaveModel(sc, model):
    '''
    存储数据
    :param sc:
    :param model:
    :return:
    '''
    try:
        model.save(sc, Path + u'data/ALSmodel')
        print('>>>>>>>> 已存储 Model 到 ALSmodel 中 <<<<<<<<')
    except Exception:
        print('>>>>>>>> Model 已经存在，请先删除再存储。 <<<<<<<<')

if __name__ == '__main__':
    sc = CreateSparkContext()
    print('=============== 数据准备阶段 ===============')
    ratingsRDD = PrepareData(sc)
    print('=============== 训练阶段 ===============')
    print('====---->>>> 开始ALS训练，参数rank=5, iterations=10, lambda=0.1 <<<<----====')
    model = ALS.train(ratingsRDD, 5, 10, 0.01)  # 经测试将iterations修改为20时执行报错，无法保存对象数据到文件
    print('=============== 存储Model ===============')
    SaveModel(sc, model)
    # try:
    #     model.save(sc, Path + u'data/ALSmodel')
    #     print('>>>>>>>> 已存储 Model 到 ALSmodel 中 <<<<<<<<')
    # except Exception:
    #     print('>>>>>>>> Model 已经存在，请先删除再存储。 <<<<<<<<')