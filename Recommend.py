#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: Recommend.py.py\n
@time: 2019-10-08 15:44\n
@desc: 读取训练好的电影推荐数据模型，并推荐用户或电影
'''
import  os, sys
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import Rating, ALS, MatrixFactorizationModel

os.environ['SPARK_HOME'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6'
os.environ['JAVA_HOME'] = 'D:\ProgramFiles\Java\jdk1.8.0_131'
os.environ['HADOOP_HOME'] = 'D:\ProgramFiles\hadoop\hadoop-2.6.4'
os.environ['PYSPARK_PYTHON'] = 'D:\ProgramFiles\Python364\python.exe'  # 在dos命令行下执行时需要，在IDE中运行配置到Run Configurations中即可，只能使用Python364中的python.exe，而Anaconda3中的python.exe报错：Cannot run program "?D:\ProgramFiles\Anaconda3\python.exe": CreateProcess error=2,

def CreateSparkContext():
    '''
    创建SparkContext对象
    :return:
    '''
    sparkConf = SparkConf().setAppName('Recommend').set('spark.ui.showConsoleProgress', 'false')
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
    print('@@@@>>>>>>>> 开始读取电影ID与名称字典... <<<<<<<<@@@@')
    itemRDD = sc.textFile(Path + u'data/ml-100k/u.item')
    movieTitle = itemRDD.map(lambda line: line.split('|')).map(lambda a: (float(a[0]), a[1])).collectAsMap()
    return (movieTitle)

def loadModel(sc):
    '''
    载入模型
    :param sc:
    :return:
    '''
    try:
        print('@@@@>>>>>>>> 载入ALSModel模型... <<<<<<<<@@@@')
        model = MatrixFactorizationModel.load(sc, Path + u'data/ALSmodel')
    except Exception:
        print('@@@@>>>>>>>> 找不到ALSModel模型，请先训练！ <<<<<<<<@@@@')
    return model

def Recommend(model):
    '''
    1、针对此用户推荐电影。eg：执行程序 Recommend.py --U 100，代表针对用户ID=100推荐电影；\n
    2、针对电影推荐给用户。eg：执行程序 Recommend.py --M 200，代表针对电影ID=200推荐给用户；
    :param model:
    :param movieTitle:
    :return:
    '''
    if sys.argv[1] == '--U':
        RecommendMovies(model, movieTitle, int(sys.argv[2]))
    if sys.argv[1] == '--M':
        RecommendUsers(model, movieTitle, int(sys.argv[2]))

def RecommendMovies(model, movieTitle, inputUserId):
    '''
    针对此用户推荐电影。eg：执行程序 Recommend.py --U 100，代表针对用户ID=100推荐电影；
    :param model:
    :param movieTitle:
    :param inputUserId:
    :return:
    '''
    RecommendMovie = model.recommendProducts(inputUserId, 10)
    print('@@@@====>>>>>>>> 针对用户 ID=' + str(inputUserId) + ' 推荐下列电影： <<<<<<<<====@@@@')
    for rmd in RecommendMovie:
        print('@@@@====>>>>>>>> 针对用户 ID={0} 推荐电影：{1} 推荐评分：{2} <<<<<<<<====@@@@'.format(rmd[0], movieTitle[rmd[1]], rmd[2]))

def RecommendUsers(model, movieTitle, inputMovieId):
    '''
    针对电影推荐给用户。eg：执行程序 Recommend.py --M 200，代表针对电影ID=200推荐给用户；
    :param model:
    :param movieTitle:
    :param inputMovieId:
    :return:
    '''
    RecommendUser = model.recommendUsers(inputMovieId, 10)
    print('@@@@====>>>>>>>> 针对电影 ID={0} 电影名：{1} 推荐下列用户： <<<<<<<<====@@@@'.format(inputMovieId, movieTitle[inputMovieId]))
    for rmd in RecommendUser:
        print('@@@@====>>>>>>>> 针对用户 ID={0} 推荐电影：{1} 推荐评分：{2} <<<<<<<<====@@@@'.format(rmd[0], movieTitle[rmd[1]], rmd[2]))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('请输入2个参数！！！')
        exit(-1)

    sc = CreateSparkContext()
    print('@@@@@@@@@@@@@@@@ 准备数据 @@@@@@@@@@@@@@@@')
    (movieTitle) = PrepareData(sc)
    print('@@@@@@@@@@@@@@@@ 载入模型 @@@@@@@@@@@@@@@@')
    model = loadModel(sc)
    print('@@@@@@@@@@@@@@@@ 进行推荐 @@@@@@@@@@@@@@@@')
    Recommend(model)

