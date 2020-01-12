# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 분류용 샘플 데이터 불러오기
iris = load_iris()
X, y, labels = iris.data, iris.target, iris.target_names

# 학습/테스트 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# 랜덤포레스트 + 그리드서치로 모델 학습
rf = RandomForestClassifier(random_state=1)
param_grid = [{'n_estimators': range(5, 50, 10), 'max_features': range(1, 4), 'max_depth': range(3, 5)}]
gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X_train, y_train)

# 그리드서치 학습 결과 출력
print('베스트 하이퍼 파라미터: {0}'.format(gs.best_params_))
print('베스트 하이퍼 파라미터 일 때 Score 점수: {0:.2f}'.format(gs.best_score_))

# 최적화 모델 추출
model = gs.best_estimator_

# 테스트세트 R^2 점수 출력
score = model.score(X_test, y_test)
print('테스트세트에서의 Score 점수: {0:.2f}'.format(score))

# 테스트세트 예측 결과 샘플 출력
predicted_y = model.predict(X_test)
for i in range(10):
    print('실제 값: {0}, 예측 값: {1}'.format(labels[y_test[i]], labels[predicted_y[i]]))
