from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss,accuracy_score
# from StringIO import StringIO
import xlwt
import xlrd

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
    dataUnion = dataUnion.fillna(0)
    # print(dataUnion.HomeTeam.unique())
    for col_name in dataUnion.columns:
        if dataUnion[col_name].dtype == 'object':
            dataUnion[col_name] = dataUnion[col_name].astype('category')
            dataUnion[col_name] = dataUnion[col_name].cat.codes

    labels = np.unique(dataUnion['HomeTeam'].values)
    # dic = {}
    # for i in range(len(labels)):
    #     dic.update({labels[i]: i})
    # df = dataUnion.replace(dic)
    #
    # ref_lables = df.Referee.unique()
    # ref_dic = {}
    # for i in range(len(ref_lables)):
    #     ref_dic.update({ref_lables[i]: i})
    # df = df.replace(ref_dic)

    Y = dataUnion['HomeTeam']
    X = dataUnion.drop(['HomeTeam'], axis='columns', inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    c_X_train, c_X_test, c_y_train, c_y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    for i in range(1, 16):
        if X_test.shape[0] == 0:
            break
        clf = DecisionTreeClassifier(max_depth=i)
        run_model(clf, X_train, y_train, X_test, y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=list(labels)))
    print(accuracy_score(true_list, pred_list))

    pred_list.clear()
    true_list.clear()
    prob_list.clear()
    for i in range(1, 16):
        if c_X_test.shape[0] == 0:
            break
        clf = KNeighborsClassifier(n_neighbors=i)
        run_model(clf, c_X_train, c_y_train, c_X_test, c_y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=list(labels)))
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

#segmentation
def dataset7():
    data = np.genfromtxt("segmentation/segmentation.test", dtype='unicode', delimiter=",")
    df = pd.DataFrame(data)

    labels = df[df.columns[0]].unique()
    dic = {}
    for i in range(len(labels)):
        dic.update({labels[i]: i})
    df = df.replace(dic)

    Y = df[df.columns[0]]
    X = df.drop(df.columns[0], axis=1, inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    c_X_train,c_X_test,c_y_train,c_y_test=X_train.copy(),X_test.copy(),y_train.copy(),y_test.copy()
    for i in range(1,16):
        if X_test.shape[0]==0:
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
        run_model(clf, c_X_train, c_y_train, c_X_test, c_y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=labels))
    print(accuracy_score(true_list, pred_list))

def dataset():

    book = xlwt.Workbook()
    ws = book.add_sheet('segmentation')  # Add a sheet
    f = open('segmentation/new1.txt', 'r+')

    data = f.readlines()  # read all lines at once
    for i in range(len(data)):
        row = data[
            i].split(",")  # This will return a line of string data, you may need to convert to other formats depending on your use case

        for j in range(len(row)):
            ws.write(i, j, row[j])  # Write to cell i, j

    book.save('segmentation/segmentation' + '.xls')
    f.close()

#healthcare
def dataset8():
    data = pd.read_csv("healthcare/healthcare-dataset-stroke-data.csv", header=0)
    df = pd.DataFrame(data)
    df.dropna(inplace= True)
    for col_name in df.columns:
        if df[col_name].dtype == 'object':
            df[col_name] = df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes

    labels = df['work_type'].unique()

    Y = df['work_type']
    X = df.drop('work_type', axis='columns', inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    c_X_train, c_X_test, c_y_train, c_y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    for i in range(1, 16):
        if X_test.shape[0] == 0:
            break
        clf = DecisionTreeClassifier(max_depth=i)
        run_model(clf, X_train, y_train, X_test, y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=list(labels)))
    print(accuracy_score(true_list, pred_list))

    pred_list.clear()
    true_list.clear()
    prob_list.clear()
    for i in range(1, 16):
        if c_X_test.shape[0] == 0:
            break
        clf = KNeighborsClassifier(n_neighbors=i)
        run_model(clf, c_X_train, c_y_train, c_X_test, c_y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=list(labels)))
    print(accuracy_score(true_list, pred_list))

#bank
def dataset10():
    data = pd.read_csv("bank/bank.csv", header=0)
    df = pd.DataFrame(data)
    df.dropna(inplace=True)
    for col_name in df.columns:
        if df[col_name].dtype == 'object':
            df[col_name] = df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes

    labels = df['poutcome'].unique()

    Y = df['poutcome']
    X = df.drop('poutcome', axis='columns', inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    c_X_train, c_X_test, c_y_train, c_y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    for i in range(1, 16):
        if X_test.shape[0] == 0:
            break
        clf = DecisionTreeClassifier(max_depth=i)
        run_model(clf, X_train, y_train, X_test, y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=list(labels)))
    print(accuracy_score(true_list, pred_list))

    pred_list.clear()
    true_list.clear()
    prob_list.clear()
    for i in range(1, 16):
        if c_X_test.shape[0] == 0:
            break
        clf = KNeighborsClassifier(n_neighbors=i)
        run_model(clf, c_X_train, c_y_train, c_X_test, c_y_test, i == 15)
    print(log_loss(true_list, prob_list, labels=list(labels)))
    print(accuracy_score(true_list, pred_list))


if __name__ == '__main__':
    # dataset1()
    # dataset2()
    # dataset3()
    # dataset4()
    # dataset5()
    # dataset6()
    # dataset7()
    # dataset()
    # dataset8()
    dataset10()

