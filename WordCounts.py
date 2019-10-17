#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: WordCounts.py\n
@time: 2019-09-29 14:50\n
@desc: 使用pyspark统计文本中单词出现的次数
'''

import  os, sys
from pyspark import SparkContext, SparkConf
os.environ['SPARK_HOME'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6'
os.environ['JAVA_HOME'] = 'D:\ProgramFiles\Java\jdk1.8.0_131'
os.environ['HADOOP_HOME'] = 'D:\ProgramFiles\hadoop\hadoop-2.6.4'
os.environ['PYSPARK_PYTHON'] = 'D:\ProgramFiles\Python364\python.exe'  # 在dos命令行下执行时需要，在IDE中运行配置到Run Configurations中即可，只能使用Python364中的python.exe，而Anaconda3中的python.exe报错：Cannot run program "?D:\ProgramFiles\Anaconda3\python.exe": CreateProcess error=2,
# os.environ['PYTHONPATH'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6\python'

# sys.path.append('D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6\python')
# sys.path.append('‪D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6\python\lib\py4j-0.10.7-src.zip')


# 如下为测试代码
# conf = SparkConf().setAppName('WordCount').setMaster('local')
# sc = SparkContext.getOrCreate(conf)
# # path = 'file:/F:/ade/test/inFile/file'
# # textFile = sc.textFile(path + '/zznueg_output_2019-07-11_1.log')
# textFile = sc.textFile('file:/F:/ade/test/inFile/file/zznueg_output_2019-07-11_1.log')
# print('success')
# print(textFile.count())
# 测试代码结束


def CreateSparkContext():
    '''
    创建SparkContext对象
    :return:
    '''
    sparkConf = SparkConf().setAppName('WordCounts').set('spark.ui.showConsoleProgress', 'false')
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
        Path = 'file:/E:/SparkPythonWorkspace/PySparkDemo/'
    else:
        Path = 'hdfs://master:9000/user/root/'


if __name__ == '__main__':
    print('开始执行---->>>>RunWordCounts')
    sc = CreateSparkContext()
    print('开始读取文本文件......')
    textFile = sc.textFile(Path + 'data/README.md')
    print('文本文件共 ' + str(textFile.count()) + ' 行')
    countsRDD = textFile.flatMap(lambda line: line.split(' ')).map(lambda x: (x, 1)).reduceByKey(lambda x, y: (x + y))
    print('文字统计共 ' + str(countsRDD.count()) + ' 项数据')
    print('开始将处理结果保存至文本文件......')
    try:
        countsRDD.saveAsTextFile(Path + 'data/output')
    except Exception as e:
        print('输出处目录已经存在，请先删除原有目录')
    sc.stop()

