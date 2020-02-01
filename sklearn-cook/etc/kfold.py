# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 분류용 샘플 데이터 불러오기
iris = load_iris()
X, y = iris.data, iris.target

# 랜덤 포레스트를 K겹 교차 검증에 사용하기
rf = RandomForestClassifier(random_state=1)
score_list = cross_val_score(rf, X, y)

# 결과 출력하기
result = list(map(lambda x: '{score:.2f}'.format(score=x), score_list))
print(result)