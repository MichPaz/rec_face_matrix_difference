# -*- coding: utf-8 -*-

# Author 1: Ernande Melo
# Author 2 and Reviser: Leonardo Monteiro

from array import array
import cv2
import pandas as pd
import numpy as np
from ia35 import machineLearning
import os.path
import os as os

# Definição de dimensões da imagem a ser classificada
H = 112
L = 92
BASE_FACES = 'att_image_base_faces/'
NUM_IMG = 8


def get_and_prepare_img(L: int, H: int) -> np.ndarray:
    img_path_test = str(input())
    img_test = cv2.imread(img_path_test)
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    img_test = cv2.resize(img_test, (L, H), interpolation=cv2.INTER_AREA)
    return img_test


def get_dif_img(img_test, L: int, H: int) -> np.ndarray:
    difx = np.ones((L, H-1))
    for i in range(0, L):
        for j in range(0, H-1):
            if (img_test[i, j] >= img_test[i][j+1]):
                difx[i][j] = (img_test[i][j] + (-1)*img_test[i][j+1])
            else:
                difx[i][j] = (img_test[i][j+1] + (-1)*img_test[i][j])
    return difx


def get_img_path_of_base(i_class: int, i_class_img: int, img_sufix=True) -> str:
    sufix = 'img' if img_sufix else ''
    args_path = [
        BASE_FACES, 's', str(i_class),
        '/', sufix, str(i_class_img), '.jpg']
    img_path_of_base = os.path.abspath(''.join(args_path))
    return img_path_of_base


def get_top_three(score_list: list):
    indexed_list = list(enumerate(score_list))
    top3 = sorted(range(len(indexed_list)),
                  key=lambda i: indexed_list[i][1], reverse=True)[:3]
    return [(x, score_list[x]) for x in top3]


def get_corr_xy(imagexf, img_test,  L: int, H: int) -> int:
    imgref = cv2.cvtColor(imagexf, cv2.COLOR_BGR2GRAY)
    det2Y = np.reshape(imgref, (H*L, 1), order='C')
    det2X = np.reshape(img_test, (H*L, 1), order='C')
    det2XX = pd.DataFrame(data=det2X)
    det2YY = pd.DataFrame(data=det2Y)
    det2XX.columns = ['A']
    det2YY.columns = ['B']
    dfres = pd.concat([det2XX, det2YY], axis=1)
    corrXY = (abs(dfres['A'].corr(dfres['B'])))
    return corrXY


img_test = get_and_prepare_img(L, H)

# Atribuições provavelmente redundantes
L = int(img_test.shape[0])
H = int(img_test.shape[1])

dif_img = get_dif_img(img_test, L, H)

res_value = []
res_index = []

path = os.listdir(BASE_FACES)
N = len(path)
n = N+1
print('number of classes', n)

for i_class in range(1, n):
    rmax = []
    for i_class_img in range(1, NUM_IMG):

        img_path_of_base = get_img_path_of_base(i_class, i_class_img)
        # print('img_path_of_base', img_path_of_base)
        img_base = cv2.imread(img_path_of_base)
        # print('img_base', img_base)

        dif_img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        detX = np.reshape(dif_img, ((H-1)*L, 1), order='C')
        detY = np.reshape(dif_img_base, ((H-1)*L, 1), order='C')
        detXX = pd.DataFrame(data=detX)
        detYY = pd.DataFrame(data=detY)
        detXX.columns = ['A']
        detYY.columns = ['B']
        dfres = pd.concat([detXX, detYY], axis=1)
        corrXY = (abs(dfres['A'].corr(dfres['B'])))

        rmax.append(corrXY)

    # print('rmax', rmax)
    max_value = max(rmax)
    max_index = rmax.index(max_value)

    # print('max_value', max_value)
    # print('max_index', max_index)
    res_value.append(max_value)
    res_index.append(max_index)

top3 = get_top_three(res_value)
print('top3', top3)

recx = max(res_value)


pstfinal = [get_img_path_of_base(x+1, res_index[x]+1, False)
            for x in [y[0] for y in top3]]


imagexf = [cv2.imread(x) for x in pstfinal]

subin = [x[:-5] for x in pstfinal]

# cv2.imwrite('img_test.jpg', img_test)

mL = int(machineLearning(subin, img_test, H, L))
print(mL)

rmax2 = [(get_corr_xy(x, img_test, L, H)) for x in imagexf]

rmx2 = max(rmax2)
result = rmax2.index(rmx2) + 1


acceptance_criteria = (rmax2[result-1] > 0.5) or (result == mL)
msg = 0 if acceptance_criteria else 1

cv2.imwrite('img00.jpg', imagexf[result-1])

arquivo = open('statusFaces.txt', 'w')
arquivo.write(str(msg)+'\n')
arquivo.close()

if (acceptance_criteria):
    arquivo = open('Id.txt', 'w')
    arquivo.write(str(subin[0])+'Id.bmp' + '\n')
    arquivo.close()
