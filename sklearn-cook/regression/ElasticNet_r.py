# -*- coding:utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import numpy as np

# 회귀용 샘플 데이터 불러오기
boston = load_boston()
X, y = boston.data, boston.target

# 학습/테스트 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 데이터 전처리(표준화, Standardization)
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

# 엘라스틱넷 + 그리드서치로 모델 학습
gb = ElasticNet(max_iter=100000)
param_grid = [{'alpha': np.linspace(0.01, 1.0, 100)}]
gs = GridSearchCV(estimator=gb, param_grid=param_grid, scoring='r2', cv=5, n_jobs=-1)
gs.fit(X_train_std, y_train)

# 그리드서치 학습 결과 출력
print('베스트 하이퍼 파라미터: {0}'.format(gs.best_params_))
print('베스트 하이퍼 파라미터 일 때 R^2 점수: {0:.2f}'.format(gs.best_score_))

# 최적화 모델 추출
model = gs.best_estimator_

# 테스트세트 R^2 점수 출력
r2_score = model.score(X_test_std, y_test)
print('테스트세트에서의 R^2 점수: {0:.2f}'.format(r2_score))

# 테스트세트 예측 결과 샘플 출력
predicted_y = model.predict(X_test_std)
for i in range(10):
    print('실제 값: {0:.2f}, 예측 값: {1:.2f}'.format(y_test[i], predicted_y[i]))
