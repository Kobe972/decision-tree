import numpy as np
from pandas import Series,DataFrame
import pickle
import math
MAX_DEPTH=20
class ClsProperty: #用于给节点分类的分类器
    """
    self.name:类型名，用于索引
    self.type:分类属性类型，0表示离散，1表示连
    self.value:属性的值
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
    def __init__(self,ctype,properties={}):
        self.properties=properties
        self.type=ctype #类型为ndarray
        self.count=len(self.properties[sorted(properties.keys())[0]]) #共有多少组数据
        assert len(self.type)==len(properties.keys())-1
        
    def add_property(self,info):
        for key,values in info:
            self.properties[key]=self.properties[key].append(info)
            
class Node:
    """
    self.data:节点存放的数据
    self.clsproperties:分类属性
    self.classifier:分类标准
    self.clsproperties:分类依据
    self.ancestor:公共祖先
    """
    def __init__(self,classes,data,depth,clsproperties=None): #classes表示会分成哪几类，data表示该节点包含的数据
        self.classes=classes
        self.num_cls=len(classes)
        self.data=data
        self.child=[]
        self.leaf=0
        self.depth=depth
        self.clsproperties=clsproperties
        if clsproperties==None:
            self.clsproperties=[]
            self.gen_clsproperties()

    def gen_clsproperties(self): #生成分类属性
        index=0
        for key,value in self.data.properties.items():
            if key=='class':
                continue
            if self.data.type[index]==0:
                self.clsproperties.append(ClsProperty(key,0,np.unique(value)))
            else:
                self.clsproperties.append(ClsProperty(key,1))
            index=index+1
            
    def entropy(self,data):#计算熵
        p=np.zeros(self.num_cls)
        for i in range(self.num_cls):
            if len(data.properties['class'])==0:
                p[i]=0
            else:
                p[i]=np.sum(data.properties['class']==self.classes[i])/data.count
                if(p[i]!=0):
                    p[i]=p[i]*math.log2(p[i])
                p[i]=-p[i]
        return p.sum()
    
    def ent_gain(self,classifier):#计算信息增益
        origin=self.entropy(self.data)
        new=0
        if classifier.type==0:#离散型变量
            for value in classifier.values:
                mask=(self.data.properties[classifier.name]==value)
                properties={}
                for key in self.data.properties.keys():
                    properties[key]=self.data.properties[key][mask]
                tmp=Data(self.data.type,properties) 
                new+=float(tmp.count)/self.data.count*self.entropy(tmp)
        else:#连续型变量
            mask=(self.data.properties[classifier.name]>=classifier.thresh)
            properties={}
            for key in self.data.properties.keys():
                properties[key]=self.data.properties[key][mask]
            tmp=Data(self.data.type,properties)
            new+=tmp.count/self.data.count*self.entropy(tmp)
            mask=(self.data.properties[classifier.name]<classifier.thresh)
            properties={}
            for key in self.data.properties.keys():
                properties[key]=self.data.properties[key][mask]
            tmp=Data(self.data.type,properties)
            new+=tmp.count/self.data.count*self.entropy(tmp)
        return origin-new
    
    def most_class(self):
        ans=[]
        for cls in self.classes:
            mask=self.data.properties['class']==cls
            ans.append(np.sum(mask))
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
        for i in range(dataset.count):
            prop={}
            for key,value in dataset.properties.items():
                prop[key]=value[i]
            prediction=self.predict(prop)
            if prediction==dataset.properties['class'][i]:
                accuracy+=1
        accuracy=accuracy/dataset.count
        return accuracy
    
    def IV(self,clsprop):
        summary=0
        if clsprop.type==0:
            for value in clsprop.values:
                mask=(self.data.properties[clsprop.name]==value)
                tmp=mask.sum()
                tmp=tmp/self.data.count
                if(tmp!=0):
                    tmp=-tmp*math.log2(tmp)
                summary+=tmp
        else:
            mask=(self.data.properties[clsprop.name]>=clsprop.thresh)
            tmp=mask.sum()
            tmp=tmp/self.data.count
            if(tmp!=0):
                summary+=-tmp*math.log2(tmp)
            tmp=1-tmp
            if(tmp!=0):
                summary+=-tmp*math.log2(tmp)
        return summary
    
    def compute_classifier(self):
        if self.depth>MAX_DEPTH:
            return None
        gain_r=np.empty(0) #信息增益率
        gain=np.empty(0) #信息增益
        classifiers=[] #候选分类方法
        print('for clsprop in self.clsproperties')
        for clsprop in self.clsproperties:
            if clsprop.type==0: #离散型数据
                _gain=self.ent_gain(clsprop)
                gain=np.append(gain,_gain)
                IV=self.IV(clsprop)
                if IV==0:
                    gain_r=np.append(gain_r,0)
                else:
                    gain_r=np.append(gain_r,_gain/self.IV(clsprop))
                classifiers.append(clsprop)
            else: #连续型数据
                print('start continuity')
                sorted_data=np.unique(self.data.properties[clsprop.name])
                if len(sorted_data)==1: #该属性的值相同
                    continue
                print('tmp=ClsProperty(clsprop.name,1)')
                tmp=ClsProperty(clsprop.name,1)
                maximum=0 #信息增益的最大值，需要调节阈值使之最大
                _gain_r=0
                print('for i in range(len(sorted_data)-1)')
                for i in range(len(sorted_data)-1): #求最合适的阈值
                    print('i=',i)
                    tmp.thresh=(float)(sorted_data[i]+sorted_data[i+1])/2 #把阈值设为相邻数据的中点
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
        if len(np.unique(self.data.properties['class']))==1: #如果节点包含数据都属于同一类，则直接把节点归为这一类
            self.leaf=1
            self.result=self.data.properties['class'][0]
            return
        if self.clsproperties==None or self.depth>MAX_DEPTH: #如果节点的分类标准用完或树超过最大深度，把节点归为数据中含有最多的那一类
            self.leaf=1
            self.result=self.most_class()
            return
        for prop in self.clsproperties: #如果样本在划分属性上取值全部相同，则把节点标记为叶节点
            if len(np.unique(self.data.properties[prop.name]))!=1:
                break
            self.leaf=1
            self.result=self.most_class()
            return
        self.classifier=self.compute_classifier() #计算分类标准
        if self.classifier==None: #如果没有合适的分类标准，把它设成子节点
            self.leaf=1
            self.result=self.most_class()
            return
        if self.classifier.type==0: #分类器离散型
            for value in self.classifier.values:
                mask=(self.data.properties[self.classifier.name]==value)
                properties={} #子节点包含的数据的属性
                for key in self.data.properties.keys():
                    properties[key]=self.data.properties[key][mask]
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
            mask=(self.data.properties[self.classifier.name]>=self.classifier.thresh)
            properties={}
            for key in self.data.properties.keys():
                properties[key]=self.data.properties[key][mask]
            new_data=Data(self.data.type,properties)
            if(new_data.count==0):
                _child=Node(self.classes,new_data,self.depth+1)
                _child.leaf=1
                self.child.append(_child)
                self.child[-1].result=self.most_class()
            else:
                mask=(self.data.properties[self.classifier.name]>=self.classifier.thresh)
                properties={}
                for key in self.data.properties.keys():
                    properties[key]=self.data.properties[key][mask]
                new_data=Data(self.data.type,properties)
                if(new_data.count==0):
                    _child=Node(self.classes,new_data,self.depth+1)
                    _child.leaf=1
                    self.child.append(_child)
                    self.child[-1].result=self.most_class()
                else:
                    self.child.append(Node(self.classes,new_data,self.depth+1,self.clsproperties.copy()))
                properties={}
                for key in self.data.properties.keys():
                    properties[key]=self.data.properties[key][~mask]
                new_data=Data(self.data.type,properties)
                if(new_data.count==0):
                    _child=Node(self.classes,new_data,self.depth+1)
                    _child.leaf=1
                    self.child.append(_child)
                    self.child[-1].result=self.most_class()
                else:
                    self.child.append(Node(self.classes,new_data,self.depth+1,self.clsproperties.copy()))
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
