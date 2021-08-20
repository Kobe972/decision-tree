from decision_tree import *
from file_utils import *

accuracy=[0,0,0,0,0]
for i in range(0,5):
    print('载入数据……')
    types,properties_train,properties_test=generate_cross_verify(4,1)
    print('处理数据……')
    train_data=Data(types,properties_train)
    test_data=Data(types,properties_test)
    tree=DTree(train_data,test_data,np.array([0,1]))
    print('开始训练……')
    tree.train()
    accuracy[i]=tree.compute_accuracy()
    print('第',i+1,'次交叉验证，精确度为',accuracy[i])
print('平均精确度为',sum(accuracy)/5)

