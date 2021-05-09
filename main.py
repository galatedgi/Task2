from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss,accuracy_score
# from StringIO import StringIO

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
      pred_list.extend(pred)
      true_list.extend(y_test_list)
      prob_list.extend(prob)
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

#frogs
def dataset1():
    data = pd.read_csv("frogs/Frogs_MFCCs.csv", header=0)
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

#covtype
def dataset2():
    data = np.genfromtxt("covtype/covtype.data", dtype=None, delimiter=",")
    df = pd.DataFrame(data)
    # print(df)
    # data=data.iloc[:, :-3]
    # df=pd.DataFrame(data)
    labels = df[df.columns[-1]].unique()
    # print(labels)
    # print(df[df.columns[-1]].value_counts())
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

#waveform
def dataset3():
    data = np.genfromtxt("waveform/waveform.data", dtype=None, delimiter=",")
    df = pd.DataFrame(data)
    # print(df)
    # data=data.iloc[:, :-3]
    # df=pd.DataFrame(data)
    labels = df[df.columns[-1]].unique()
    # print(labels)
    # print(df[df.columns[-1]].value_counts())
    Y = df[df.columns[-1]]
    # columns = df.columns.tolist()  # get the columns
    # cols_to_use = columns[:len(columns) - 1]
    X = df.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    c_X_train, c_X_test, c_y_train, c_y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    for i in range(1, 16):
        if X_test.shape[0] == 0:
            break
        clf = DecisionTreeClassifier(max_depth=i)
        run_model(clf, X_train, y_train, X_test, y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=labels))
    print(accuracy_score(true_list, pred_list))

    pred_list.clear()
    true_list.clear()
    prob_list.clear()
    for i in range(1, 16):
        if c_X_test.shape[0] == 0:
            break
        clf = KNeighborsClassifier(n_neighbors=i)
        run_model(clf,c_X_train,c_y_train,c_X_test,c_y_test,i==15)
    print(log_loss(true_list,prob_list,labels=labels))
    print(accuracy_score(true_list,pred_list))

#Dry_Bean
def dataset4():
    df = pd.read_csv("DryBeanDataset/Dry_Bean.csv", header=0)
    # data=data.iloc[:, :-3]
    # df=pd.DataFrame(data)
    labels=df.Class.unique()
    dic={}
    for i in range(len(labels)):
        dic.update({labels[i]:i})
    df=df.replace(dic)
    print(df.Class.unique())
    print(df['Class'].value_counts())
    Y = df['Class']
    X = df.drop('Class',axis='columns', inplace=False)
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

#Football
def dataset5():
    data1 = pd.read_csv("Football/season-1011_csv.csv", header=0)
    data2 = pd.read_csv("Football/season-1112_csv.csv", header=0)
    data3 = pd.read_csv("Football/season-1213_csv.csv", header=0)
    data4 = pd.read_csv("Football/season-1314_csv.csv", header=0)
    data5 = pd.read_csv("Football/season-0910_csv.csv", header=0)

    dataUnion = pd.concat([data1, data2, data3, data4, data5], ignore_index=True)
    # dataUnion.isnull().any()
    # print(dataUnion.isnull().sum())
    dataUnion = dataUnion.fillna(0)
    # print(dataUnion)
    print(dataUnion.HomeTeam.unique())
    # labels = np.unique(dataUnion[['HomeTeam', 'AwayTeam', 'Referee']].values)
    labels = np.unique(dataUnion['HomeTeam'].values)
    # labels = dataUnion.HomeTeam.unique()
    dic = {}
    for i in range(len(labels)):
        dic.update({labels[i]: i})
    df = dataUnion.replace(dic)

    # away_team_labels = dataUnion.AwayTeam.unique()
    # away_team_dic = {}
    # for i in range(len(away_team_labels)):
    #     away_team_dic.update({away_team_labels[i]: i})
    # df = dataUnion.replace(away_team_dic)
    # print(df.dtypes)

    # df['Referee'] = df['Referee'].astype(int)
    ref_lables = df.Referee.unique()
    ref_dic = {}
    for i in range(len(ref_lables)):
        ref_dic.update({ref_lables[i]: i})
    df = df.replace(ref_dic)

    # print(df.HomeTeam.unique())
    # print(df['HomeTeam'].value_counts())
    # df = df.drop(['Date', 'Div'], axis='columns', inplace=False)
    Y = df['HomeTeam']
    X = df.drop(['HomeTeam', 'Date', 'Div', 'HTR', 'FTR'], axis='columns', inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    c_X_train, c_X_test, c_y_train, c_y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    for i in range(1, 16):
        if X_test.shape[0] == 0:
            break
        clf = DecisionTreeClassifier(max_depth=i)
        run_model(clf, X_train, y_train, X_test, y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=list(dic.values())))
    print(accuracy_score(true_list, pred_list))

    pred_list.clear()
    true_list.clear()
    prob_list.clear()
    for i in range(1, 16):
        if c_X_test.shape[0] == 0:
            break
        clf = KNeighborsClassifier(n_neighbors=i)
        run_model(clf, c_X_train, c_y_train, c_X_test, c_y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=list(dic.values())))
    print(accuracy_score(true_list, pred_list))

#avila
def dataset6():
    data_train = np.genfromtxt("avila/avila-tr.txt", dtype='unicode', delimiter=",")
    # data_test = np.genfromtxt("avila/avila-ts.txt", dtype='unicode', delimiter=",")
    # dataUnion = np.concatenate((data_train, data_test), axis=0)
    df = pd.DataFrame(data_train)

    labels = df[df.columns[-1]].unique()
    dic = {}
    for i in range(len(labels)):
        dic.update({labels[i]: i})
    df = df.replace(dic)
    # print(labels)
    # print(df[df.columns[-1]].value_counts())
    Y = df[df.columns[-1]]
    X = df.iloc[: , :-1]
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
    # dataset2()
    # dataset3()
    # dataset4()
    # dataset5()
    dataset6()


