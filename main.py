from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss,accuracy_score


pred_list=[]
true_list=[]
prob_list=[]

def run_model(clf,x_train,y_train,x_test,y_test,condition):
  clf.fit(x_train,y_train)
  pred=clf.predict(x_test)
  prob=clf.predict_proba(x_test)
  index_to_remove=[]
  y_test_list=list(y_test)
  if condition:
      pred_list.append(*pred)
      true_list.append(*y_test_list)
      prob_list.append(*prob)
      return

  for i in range(pred.shape[0]):
    if prob[i][np.where(clf.classes_==pred[i])] > 0.95:
      pred_list.append(pred[i])
      true_list.append(y_test_list[i])
      prob_list.append(prob[i])
      index_to_remove.append(i)

  # score=clf.score(np.array(pred_test)[indexes],np.array(y_test)[indexes])
  # f_score=f1_score(np.array(y_test)[indexes],np.array(pred_test)[indexes])
  if not index_to_remove:
      return

  # score=log_loss(np.array(y_test_list)[index_to_remove],prob[index_to_remove],labels=labels)
  x_test.drop(x_test.index[index_to_remove], inplace=True)
  y_test.drop(y_test.index[index_to_remove], inplace=True)

  return

def dataset1():
    data = pd.read_csv("Frogs_MFCCs.csv", header=0)
    data=data.iloc[:, :-3]
    df=pd.DataFrame(data)
    labels=df.Family.unique()
    dic={}
    for i in range(len(labels)):
        dic.update({labels[i]:i})
    df=df.replace(dic)
    print(df.Family.unique())
    print(df['Family'].value_counts())
    Y = df['Family']
    X = df.drop('Family',axis='columns', inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    c_X_train,c_X_test,c_y_train,c_y_test=X_train.copy(),X_test.copy(),y_train.copy(),y_test.copy()
    for i in range(1,16):
        if X_test.shape[0]==0:
            break
        clf = DecisionTreeClassifier(max_depth=i)
        run_model(clf,X_train, y_train, X_test, y_test,i==15)
    print(log_loss(true_list,prob_list,labels=list(dic.values())))
    print(accuracy_score(true_list,pred_list))

    pred_list.clear()
    true_list.clear()
    prob_list.clear()
    for i in range(1,16):
        if c_X_test.shape[0]==0:
            break
        clf = KNeighborsClassifier(n_neighbors=i)
        run_model(clf,c_X_train,c_y_train,c_X_test,c_y_test,i==15)
    print(log_loss(true_list,prob_list,labels=list(dic.values())))
    print(accuracy_score(true_list,pred_list))

def dataset2():
    data = np.genfromtxt("covtype.data", dtype=None, delimiter=",")
    df = pd.DataFrame(data)
    print(df)
    # data=data.iloc[:, :-3]
    # df=pd.DataFrame(data)
    labels = df[df.columns[-1]].unique()
    print(labels)
    print(df[df.columns[-1]].value_counts())
    Y = df[df.columns[-1]]
    X = df.drop(df[df.columns[-1]], axis='columns', inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    c_X_train,c_X_test,c_y_train,c_y_test=X_train.copy(),X_test.copy(),y_train.copy(),y_test.copy()
    for i in range(1,16):
        if X_test.shape[0]==0:
            break
        clf = DecisionTreeClassifier(max_depth=i)
        run_model(clf,X_train, y_train, X_test, y_test,i==15)
    print(log_loss(true_list,prob_list,labels=labels))
    print(accuracy_score(true_list,pred_list))

    pred_list.clear()
    true_list.clear()
    prob_list.clear()
    for i in range(1,16):
        if c_X_test.shape[0]==0:
            break
        clf = KNeighborsClassifier(n_neighbors=i)
        run_model(clf,c_X_train,c_y_train,c_X_test,c_y_test,i==15)
    print(log_loss(true_list,prob_list,labels=labels))
    print(accuracy_score(true_list,pred_list))

if __name__ == '__main__':
    # dataset1()
    dataset2()
