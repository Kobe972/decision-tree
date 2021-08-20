import numpy as np
from pandas import Series,DataFrame
import pickle
import math
from pyspark.sql import SparkSession
from pyspark.sql import Row
import pyspark.sql.functions as fn
from pyspark.sql.types import *
import pandas as pd
import time

MAX_DEPTH=20
class ClsProperty: #用于给节点分类的分类器
    """
    self.name:类型名，用于索引
    self.type:分类属性类型，0表示离散，1表示连
    self.values:属性的值
    self.thresh:连续属性分类阈值
    """
    def __init__(self,name,ctype,values=np.empty(0)):
        self.name=name
        self.type=ctype #类型为int32
        self.values=values
    def copy(self):
        ret=ClsProperty(self.name,self.type,self.values)
        if self.type==1:
            ret.thresh=self.thresh
        return ret

        
class Data:
    def __init__(self,ctype,properties):
        self.properties=properties #数据属性，类型为pyspark dataframe，列名为属性名称，每行对应一个数据，标记分类的列名为'class'
        self.type=ctype #类型为ndarray
        self.count=self.properties.count() #共有多少组数据
        #assert len(self.type)==len(self.properties.columns)-1
            
class Node:
    """
    self.data:节点存放的数据，为Data类型
    self.clsproperties:分类属性，为列表，元素是Clsproperty类
    self.classifier:分类标准，是Clsproperty类，需要根据数据计算
    self.ancestor:公共祖先
    self.leaf:是否为叶子节点
    """
    def __init__(self,classes,data,depth,clsproperties=None): #classes表示会分成哪几类，为numpy数组，元素类型任选，通常是string；data表示该节点包含的数据，depth为当前深度
        self.classes=classes
        self.num_cls=len(classes)
        self.data=data
        self.child=[]
        self.leaf=0
        self.time_list=np.empty(0)
        self.depth=depth
        self.clsproperties=clsproperties
        if clsproperties==None:
            self.clsproperties=[]
            self.gen_clsproperties()

    def gen_clsproperties(self): #生成分类属性
        index=0
        for column in self.data.properties.columns:
            if index==0:
                index+=1
                continue
            if self.data.type[index-1]==0:
                self.clsproperties.append(ClsProperty(str(column),0,np.array(self.data.properties.select(str(column)).distinct().collect())))
            else:
                self.clsproperties.append(ClsProperty(str(column),1))
            index+=1
            
    def entropy(self,data,count):#计算熵
        data=data.groupby('class').count()
        ret=data.select(fn.sum(-data['count']/count*fn.log2(data['count']/count))).head()[0]
        return ret
    
    def ent_gain(self,classifier):#计算信息增益
        origin=self.entropy(self.data.properties,self.data.count)
        new=0
        if classifier.type==0:#离散型变量
            for value in classifier.values:
                properties=self.data.properties.filter(self.data.properties[classifier.name]==value[0]).select('class')
                tmp=properties.count() 
                new+=float(tmp)/self.data.count*self.entropy(properties,tmp)
        else:#连续型变量
            properties=self.data.properties.filter(self.data.properties[classifier.name]>=classifier.thresh).select('class')
            tmp=properties.count()
            new+=float(tmp)/self.data.count*self.entropy(properties,tmp)
            properties=self.data.properties.filter(self.data.properties[classifier.name]<classifier.thresh).select('class')
            tmp=properties.count()
            new+=float(tmp)/self.data.count*self.entropy(properties,tmp)
        return origin-new
    
    def most_class(self): #返回节点数据属于哪一类的居多
        ans=[]
        for cls in self.classes:
            ans.append(self.data.properties.filter(self.data.properties['class']==str(cls)).count())
        return self.classes[np.argmax(ans)]
    
    def predict(self,prop): #根据属性prop预测结果
        if self.leaf==1:
            return self.result
        if self.classifier.type==0:
            for i in range(len(self.classifier.values)):
                if prop[self.classifier.name]==self.classifier.values[i]:
                    return self.child[i].predict(prop)
        else:
            if prop[self.classifier.name]>=self.classifier.thresh:
                return self.child[0].predict(prop)
            else:
                return self.child[1].predict(prop)
    def compute_accuracy(self,dataset): #在dataset数据集（类型为Data）上计算分类的精确度
        accuracy=0
        collection=dataset.properties.collect()
        for i in range(dataset.count):
            prop={}
            for key in dataset.properties.columns:
                prop[key]=collection[i][key]
            prediction=self.predict(prop)
            if str(prediction)==str(collection[i]['class']):
                accuracy+=1
        accuracy=accuracy/dataset.count
        return accuracy
    
    def IV(self,clsprop):
        summary=0
        if clsprop.type==0:
            for value in clsprop.values:
                tmp=self.data.properties.filter(self.data.properties[clsprop.name]==value[0]).count()/self.data.count
                if(tmp!=0):
                    tmp=-tmp*math.log2(tmp)
                summary+=tmp
        else:
            tmp=self.data.properties.filter(self.data.properties[clsprop.name]>=clsprop.thresh).count()/self.data.count
            if(tmp!=0):
                summary+=-tmp*math.log2(tmp)
            if(tmp!=1):
                summary+=-(1-tmp)*math.log2(1-tmp)
        return summary
    
    def compute_classifier(self):
        if self.depth>MAX_DEPTH:
            return None
        gain_r=np.empty(0) #信息增益率
        gain=np.empty(0) #信息增益
        classifiers=[] #候选分类方法
        print('for clsprop in self.clsproperties')
        for clsprop in self.clsproperties:
            print('clsprop:',clsprop.name)
            if clsprop.type==0: #离散型数据
                _gain=self.ent_gain(clsprop)
                gain=np.append(gain,_gain)
                IV=self.IV(clsprop)
                if IV==0:
                    gain_r=np.append(gain_r,0)
                else:
                    gain_r=np.append(gain_r,_gain/IV)
                classifiers.append(clsprop)
            else: #连续型数据
                sorted_data=self.data.properties.select(clsprop.name).distinct().sort(clsprop.name).collect()
                if len(sorted_data)==1: #该属性的值相同
                    continue
                tmp=ClsProperty(clsprop.name,1)
                maximum=0 #信息增益的最大值，需要调节阈值使之最大
                _gain_r=0
                print('len(sorted_data)=',len(sorted_data))
                start_time=time.time()
                for i in range(len(sorted_data)-1): #求最合适的阈值
                    print('i=',i)
                    if i!=0:
                        self.time_list=np.append(self.time_list,time.time()-start_time)
                        start_time=time.time()
                        print('T:',self.time_list[-1])
                    if i>=30:
                        print('avg:',self.time_list.mean())
                    tmp.thresh=(float)(sorted_data[i][clsprop.name]+sorted_data[i+1][clsprop.name])/2 #把阈值设为相邻数据的中点
                    new_gain=self.ent_gain(tmp)
                    if maximum<=new_gain: #更新分类器
                        chosen=tmp.copy()
                        _gain_r=new_gain/self.IV(tmp) 
                        maximum=new_gain
                gain=np.append(gain,maximum)
                classifiers.append(chosen)
                gain_r=np.append(gain_r,_gain_r)
        if len(classifiers)==0: #如果没找到合适的分类器，返回None
            return None
        m=gain.mean()
        maximum=0
        ret=classifiers[0]
        for i in range(len(gain_r)):
            if gain_r[i] >= maximum:
                if gain[i]>=m:
                    maximum=gain_r[i]
                    ret=classifiers[i]
        return ret #返回信息增益大于等于平均水平的分类器中信息增益率最大的
    
    def train(self): #训练
        if self.leaf==1: #如果已经是叶节点，直接返回
            return
        if self.data.properties.select('class').distinct().count()==1: #如果节点包含数据都属于同一类，则直接把节点归为这一类
            self.leaf=1
            self.result=self.data.properties.select('class').distinct().take(1)[0]['class']
            return
        if self.clsproperties==None or self.depth>MAX_DEPTH: #如果节点的分类标准用完或树超过最大深度，把节点归为数据中含有最多的那一类
            self.leaf=1
            self.result=self.most_class()
            return
        for prop in self.clsproperties: #如果样本在划分属性上取值全部相同，则把节点标记为叶节点
            if self.data.properties.select(prop.name).distinct().count()!=1:
                break
            self.leaf=1
            self.result=self.most_class()
            return
        print('start self.compute_classifier()')
        self.classifier=self.compute_classifier() #计算分类标准
        print('end self.compute_classifier()')
        if self.classifier==None: #如果没有合适的分类标准，把它设成子节点
            self.leaf=1
            self.result=self.most_class()
            return
        if self.classifier.type==0: #分类器离散型
            for value in self.classifier.values:
                properties=self.data.properties.filter(self.data.properties[self.classifier.name]==value[0])
                new_data=Data(self.data.type,properties) #子节点包含的数据
                if(new_data.count==0): #子节点已经不包含任何数据，将其设为叶节点，类型为父节点所带数据中占比最大的类一类
                    _child=Node(self.classes,new_data,self.depth+1)
                    _child.leaf=1
                    self.child.append(_child)
                    self.child[-1].result=self.most_class()
                else:
                    _clsproperties=self.clsproperties.copy()
                    _clsproperties.remove(self.classifier) #删去该节点的分类器，作为子节点的分类标准集合
                    self.child.append(Node(self.classes,new_data,self.depth+1,_clsproperties))
        else: #分类器为连续型
            properties=self.data.properties.filter(self.data.properties[self.classifier.name]>=self.classifier.thresh)
            new_data=Data(self.data.type,properties)
            if(new_data.count==0):
                _child=Node(self.classes,new_data,self.depth+1)
                _child.leaf=1
                self.child.append(_child)
                self.child[-1].result=self.most_class()
            else:
                self.child.append(Node(self.classes,new_data,self.depth+1,self.clsproperties.copy()))
                properties=self.data.properties.filter(self.data.properties[self.classifier.name]<self.classifier.thresh)
                new_data=Data(self.data.type,properties)
                if(new_data.count==0):
                    _child=Node(self.classes,new_data,self.depth+1)
                    _child.leaf=1
                    self.child.append(_child)
                    self.child[-1].result=self.most_class()
                else:
                    self.child.append(Node(self.classes,new_data,self.depth+1,self.clsproperties.copy()))
        for _child in self.child:
            _child.data.properties.persist()
        self.data.properties.unpersist()
        for _child in self.child:
            _child.train()
        return
                
                
class DTree:
    def __init__(self,train_data,test_data,classes):
        self.train_data=train_data
        self.test_data=test_data
        self.classes=classes
        self.ancestor=Node(classes,train_data,0)
        self.ancestor.test_data=test_data
    def train(self,prune=True):
        self.ancestor.train()
        if prune:
            self.prune(self.ancestor)
    def compute_accuracy(self):
        return self.ancestor.compute_accuracy(self.test_data)
    def predict(self,prop):
        return self.ancestor.predict(prop)
    def predict(self,prop):
        _prop={}
        index=0
        for key in self.train_data.properties.keys():
            if key=='class':
                continue
            _prop[key]=prop[index]
            index+=1
        return self.ancestor.predict(_prop)
    def prune(self,node): #剪枝
        if node.leaf==1:
            return
        for child in node.child:
            self.prune(child)
        origin_accuracy=self.compute_accuracy()
        node.leaf=1
        node.result=node.most_class()
        new_accuracy=self.compute_accuracy()
        if new_accuracy>=origin_accuracy: #如果去掉子节点后泛化性能加强，则去掉子节点
            node.child=[]
            return
        node.leaf=0
