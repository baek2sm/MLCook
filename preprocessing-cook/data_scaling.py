# -*- coding:utf-8 -*-
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import numpy as np


# 데이터의 평균 및 표준편차 출력
def scaling_and_printing(scaler):
    global X_train, X_test
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    print('학습 데이터 평균: {mean:.2f}, 분산: {std:.2f}'.format(mean=np.mean(X_train_std), std=np.std(X_train_std)))
    print('테스트 데이터 평균: {mean:.2f}, 분산: {std:.2f}\n'.format(mean=np.mean(X_test_std), std=np.std(X_test_std)))


# 데이터셋 불러오기
dataset = load_breast_cancer()
X, y, labels = dataset.data, dataset.target, dataset.target_names

# 학습, 테스트 데이터세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

# StandardScaler를 이용한 데이터 스케일 조정
print('StandardScaler 적용 결과')
scaling_and_printing(StandardScaler())

# RobustScaler를 이용한 데이터 스케일 조정
print('RobustScaler 적용 결과')
scaling_and_printing(RobustScaler())

# MinMaxScaler를 이용한 데이터 스케일 조정
print('MinMaxScaler 적용 결과')
scaling_and_printing(MinMaxScaler())

# Normalizer를 이용한 데이터 스케일 조정
print('Normalizer 적용 결과')
scaling_and_printing(Normalizer())