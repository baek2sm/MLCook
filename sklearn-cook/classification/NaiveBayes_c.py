# -*- coding:utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import numpy as np

# 분류용 샘플 데이터 불러오기
news = fetch_20newsgroups()
X, y, labels = news.data, news.target, news.target_names

# 학습/테스트 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# 데이터 전처리(벡터화)
vectorizer = CountVectorizer()
tfid = TfidfTransformer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

X_train_tfid = tfid.fit_transform(X_train_vec)
X_test_tfid = tfid.transform(X_test_vec)

# 다중분류 나이브 베이브 + 그리드서치로 모델 학습
nb = MultinomialNB()
param_grid = [{'alpha': np.linspace(0.01, 1, 100)}]
gs = GridSearchCV(estimator=nb, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X_train_tfid, y_train)

# 그리드서치 학습 결과 출력
print('베스트 하이퍼 파라미터: {0}'.format(gs.best_params_))
print('베스트 하이퍼 파라미터 일 때 정확도: {0:.2f}'.format(gs.best_score_))

# 최적화 모델 추출
model = gs.best_estimator_

# 테스트세트 정확도 출력
score = model.score(X_test_tfid, y_test)
print('테스트세트에서의 정확도: {0:.2f}'.format(score))

# 테스트세트 예측 결과 샘플 출력
predicted_y = model.predict(X_test_tfid)
for i in range(10):
    print('실제 값: {0}, 예측 값: {1}'.format(labels[y_test[i]], labels[predicted_y[i]]))
