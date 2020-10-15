import pandas as pd
from sklearn.datasets import load_iris
iris_data = load_iris()

pandas_DF = pd.DataFrame(data=iris_data.data, columns=['sepal_length','sepal_width','petal_length','petal_width']) #
# print(pandas_DF)  #데이터 전체적으로 보여준다 젤 앞 줄은 인덱스로 판다스가 자체적으로 붙여주는거
# print(type(pandas_DF))
#
print(pandas_DF.head(10)) # 위에서 부터 10개 보여줘
# print(pandas_DF.shape) # 150줄 4개 칼럼
# #
# print(pandas_DF.info()) # 각컬럼의 형태
# print(pandas_DF.describe()) #각 컬럼의 기초통계량 카운트, 평균, 분산 등등
#
# print(pandas_DF['sepal_width'].value_counts()) #칼럼의 데이터 타입, 동일한 값이 몇 번 반복되는지
#
# year_feature = pandas_DF['sepal_width']
# print(year_feature.head(10))
# year_value = pandas_DF['sepal_width'].value_counts()
# print(year_value)

# print(pandas_DF.columns) #데이터가 가진 컬럼 뭐뭐 있는지