# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
from pprint import pprint



data_train=pd.read_csv(r'C:\Users\Mohammad\Downloads\Adult\train.discrete',
                       names=['50k','workclass','education','martial-ststus','occupation',
                              'relationship','race','sex','native-country'])
data_test=pd.read_csv(r'C:\Users\Mohammad\Downloads\Adult\test.discrete',
                      names=['50k','workclass','education','martial-ststus','occupation',
                             'relationship','race','sex','native-country'])


data_train=data_train.rename(columns={'50k':'label'})
data_test=data_test.rename(columns={'50k':'label'})
cols=['workclass','education','martial-ststus','occupation',
       'relationship','race','sex','native-country','lable']



def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts))
for i in range(len(elements))])
    return entropy

def InfoGain(data,split_attribute_name,target_name="label"):
    total_entropy = entropy(data[target_name])
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain



def ID3(data,originaldata,features,target_attribute_name="label",parent_node_class = None):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
        
    elif len(features) ==0:
        return parent_node_class
    
    
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] 
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature:{}}
        
        
        features = [i for i in features if i != best_feature]
        
        
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            
            subtree = ID3(sub_data,data_train,features,target_attribute_name,parent_node_class)
            
            tree[best_feature][value] = subtree
            
        return(tree)

def predict(query,tree,default = 1):

    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result


def test(data,tree):
    queries = data.iloc[:,1:].to_dict(orient = "records")
    
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    #print('The test accuracy is: ',(np.sum(predicted["predicted"] == data.reset_index(drop=True)["label"])/len(data))*100,'%')
    acc=(np.sum(predicted["predicted"] == data.reset_index(drop=True)["label"])/len(data))*100
    return acc


tree = ID3(data_train,data_train,data_train.columns[1:])
print('Train accuracy is:',test(data_train,tree),'%')
print('Test accuracy is:',test(data_test,tree),'%')
print('Tree size is:',len(tree[list(tree.keys())[0]])+1)





#1_a


train_acc=[]
test_acc=[]
tree_size=[]


for i in range(3):
  idx = np.random.choice(len(data_train),size=int(0.3*len(data_train)))
  new_data=data_train.iloc[idx,:]
  tree = ID3(new_data,new_data,new_data.columns[1:])
  train_acc.append(test(new_data,tree))
  test_acc.append(test(data_test,tree))
  tree_size.append(len(tree[list(tree.keys())[0]])+1)
  

print('Mean accuracy of train data is:',np.mean(train_acc),'%')
print('Mean accuracy of test data is:',np.mean(test_acc),'%')
print('Mean size of tree is:',np.mean(tree_size))
print("finish30%")


#1_b

P=[0.4,0.5,0.6,0.7,1]
result=[]

for percent in P:
  train_acc=[]
  test_acc=[]
  tree_size=[]
  for i in range(3):
     idx = np.random.choice(len(data_train),size=int(percent*len(data_train)))
     new_data=data_train.iloc[idx,:]
     tree = ID3(new_data,new_data,new_data.columns[1:])
     train_acc.append(test(new_data,tree))
     test_acc.append(test(data_test,tree))
     tree_size.append(len(tree[list(tree.keys())[0]])+1)
  result.append([percent,np.mean(train_acc),np.mean(test_acc),np.mean(tree_size)])

cols = ["Percent of train dataset", "Train accuracy", "Test accuracy",'Tree size']
df = pd.DataFrame(result , columns=cols)
print(df)
print("\n \n")
print(df.describe())
print("finish")
#2_a
from sklearn.model_selection import train_test_split
result=[]
col=data_train.columns
for i in range(3,len(col)):
  data_train_2, data_val, _, _ = train_test_split(data_train, data_train, test_size=0.25)
  data_train_2=data_train_2[col[:i]]
  data_val=data_val[col[:i]]
  data_test_new=data_test[col[:i]]
  tree = ID3(data_train_2,data_train_2,data_train_2.columns[1:])
  result.append([i-1,test(data_train_2,tree),test(data_val,tree),test(data_test_new,tree)])

df = pd.DataFrame(result , columns=['Node','Train accuracy','Test accuracy','Val accuracy'])
df.index=df['Node']
df[['Train accuracy','Test accuracy','Val accuracy']].plot()



#2_b
result=[]
col=data_train.columns
for i in range(3,len(col)):
  data_test_new, data_val, _, _ = train_test_split(data_test, data_test, test_size=0.25)
  data_train_2=data_train[col[:i]]
  data_val=data_val[col[:i]]
  data_test_new=data_test_new[col[:i]]
  tree = ID3(data_train_2,data_train_2,data_train_2.columns[1:])
  result.append([i-1,test(data_train_2,tree),test(data_val,tree),test(data_test_new,tree)])

df = pd.DataFrame(result , columns=['Node','Train accuracy','Test accuracy','Val accuracy'])
df.index=df['Node']
df[['Train accuracy','Test accuracy','Val accuracy']].plot()



