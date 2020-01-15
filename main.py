# import testcase
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

def myIrisDo():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    # X = iris.data[:, [0, 1]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)

    # X_combined_std = np.vstack((X_train_std, X_test_std))
    # y_combined = np.hstack((y_train, y_test))

    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train, y_train)
    # print(">>>>X_test_std[0, :]")
    # print(X_test_std[0, :])
    # print(X_test_std[:, 0])
    # print(X_test_std[:, 1])

    y_pred = lr.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(">> cnf_matrix")
    print(cnf_matrix)

    print(">> accuracy_score")
    print(lr.score(X_test, y_test))
    print(accuracy_score(y_test, y_pred))

    print(">> params")
    print(lr.get_params())

    print(">> coef")
    print(lr.coef_)

    print(">> intercept_")
    print(lr.intercept_)

    lr.predict_proba(X_test)  # 查看第一个测试样本属于各个类别的概率

    # lr.predict_proba(X_test_std[0, :])  # 查看第一个测试样本属于各个类别的概率
    scatter_highlight_kwargs = {'s': 60, 'label': 'Test data', 'alpha': 0.7}
    scatter_kwargs = {'s': 100, 'edgecolor': None, 'alpha': 0.7}
    plot_decision_regions(X_train, y_train, X_highlight=X_test, clf=lr, legend=2,
                          scatter_kwargs=scatter_kwargs,
                          scatter_highlight_kwargs=scatter_highlight_kwargs)  # test_idx=range(105, 150) test_idx=range(75, 150)
    # plot_decision_regions(X_test, y_test, clf=lr)

    # plot_decision_regions(X_test, y_pred, clf=lr)

    # plot_decision_regions(X_test_std, y_pred, classifier=lr, test_idx=range(0, 75))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    # irisDo()
    myIrisDo()

    # e = testcase.readExpect()
    # i = testcase.readInputs()
    # print(e, "\n")
    # print(i, "\n")
    # iris = datasets.load_iris()
    #
    # # data对应了样本的4个特征，150行4列
    # print('>> shape of data:')
    # print(iris.data.shape)
    #
    # # 显示样本特征的前5行
    # print('>> line top 5:')
    # # print(iris.data[:5])
    # print(iris.data)
    #
    # print('>> 2,3 ')
    # print(iris.data[:, [2, 3]])
    #
    # # target对应了样本的类别（目标属性），150行1列
    # print('>> shape of target:')
    # print(iris.target.shape)
    #
    # # 显示所有样本的目标属性
    # print('>> show target of data:')
    # print(iris.target)
    #
