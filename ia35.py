# -*- coding: utf-8 -*-

# Author 1: Ernande Melo
# Author 2 and Reviser: Leonardo Monteiro

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2

NUM_IMG = 8


def machineLearning(F, X, H, L):
    Matt = [np.zeros(((NUM_IMG), H*L), dtype=np.float) for x in F]
    Inx = np.zeros((H*L, 1), dtype=np.int)
    In = [np.zeros((H*L, 1), dtype=np.int) for x in F]

    R = []
    f_index = 0
    for f in F:
        R.append(0)
        for cnj in range(1, NUM_IMG):
            pstx0 = str(f)
            pstx2 = str(pstx0) + str(cnj)
            pstx4 = str(pstx2)+'.jpg'
            image = cv2.imread(pstx4)
            imgtstpi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imgtstpi = cv2.resize(
                imgtstpi, (L, H), interpolation=cv2.INTER_AREA)
            In = np.reshape(imgtstpi, (H*L, 1))
            for i in range(0, H*L):
                Matt[f_index][R[f_index]][i] = In[i]
            R[f_index] += 1
        f_index += 1

    kclass = np.zeros((len(F)*(NUM_IMG), 1), dtype=np.float)
    dF = [(pd.DataFrame(data=Matt[x])) for x in range(len(F))]
    u = 0
    for j in range(0, (NUM_IMG)*3):
        if ((j % (NUM_IMG)*3) == 0):
            u = u+1
        kclass[j] = u
    # print(kclass)
    dFs = pd.concat(dF, axis=0).reset_index(drop=True)
    data_table0 = pd.DataFrame(data=dFs)
    data_table0['kclass'] = kclass
    y = data_table0['kclass']
    x = data_table0.drop(['kclass'], axis=1)
    # print(data_table0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    knn = KNeighborsClassifier(n_neighbors=1)
    # Treino
    knn.fit(x_train, y_train)
    # Teste
    previsaoKNN = knn.predict(x_test)
    Kscore = knn.score(x_test, y_test.values.ravel())
    Kacc = accuracy_score(y_test, previsaoKNN)
    Outknn = ['Score:', Kscore, 'Accuracy: ', Kacc]
    print(Outknn)
    print(confusion_matrix(y_test, previsaoKNN))
    # print(x_test)
    Inx = np.reshape(X, (1, H*L))
    Xx = pd.DataFrame(data=Inx)
    KX = (knn.predict(Xx.values))
    return (KX)
