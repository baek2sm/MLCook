from sklearn.preprocessing import OneHotEncoder
import pandas as pd

data_dic = {'label': ['Apple', 'Samsung', 'LG', 'Samsung']}
df = pd.DataFrame(data_dic)
pd_oh_encoded = pd.get_dummies(df['label'])
print(pd_oh_encoded)

oh = OneHotEncoder()
sk_oh_encoded = oh.fit_transform(df)
print(sk_oh_encoded)

