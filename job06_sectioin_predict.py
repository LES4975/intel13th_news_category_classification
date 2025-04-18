import pickle
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from konlpy.tag import Okt
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re

# 데이터셋 정리하기
df = pd.read_csv('./crawling_data/naver_headline_news_20250418.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.head())
df.info()
print(df.category.value_counts())

# 전처리
X = df.titles
Y = df.category

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)
label = encoder.classes_
print(label)

# One-Hot Encoding
labeled_y = encoder.transform(Y) # 라벨링
onehot_y = to_categorical(labeled_y)
print(onehot_y)

# 형태소 분리
okt = Okt()
for i in range(len(X)):
    X[i] = re.sub('[^가-힣]', ' ', X[i])  # 모든 문장에 대해 한글만 남기기
    X[i] = okt.morphs(X[i], stem=True)
print(X)

# 한 글자 단위의 형태소 제거하기
for idx, sentence in enumerate(X):
    words = []
    for word in sentence:
        if len(word) > 1:
            words.append(word) # 형태소의 길이가 1보다 큰 것만 words에 추가
    X[idx] = ' '.join(words) # 형태소들을 이어 붙인다(사이에 공백을 넣음)
print(X[:10])

# 토큰화
with open('./models/token_max_25.pickle', 'rb') as f: # 저번에 만든 토큰(max=25) 불러 오기
    token = pickle.load(f)
tokened_x = token.texts_to_sequences(X)
print(tokened_x[:5])

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 25: # 토큰화한 데이터 중에서 길이가 max값보다 긴 것이 있다면
        tokened_x[i] = tokened_x[i][:25] # 25번째까지의 토큰만 저장한다.

x_pad = pad_sequences(tokened_x, 25)
print(x_pad)

# 예측
model = load_model('./models/news_section_classification_model_0.7273651957511902.h5')
preds = model.predict(x_pad)
print(preds)

# 예측 결과 가장 가능성이 높은 카테고리를 순서대로 리스트에 저장
predict_section = []
for pred in preds:
    predict_section.append(label[np.argmax(pred)])
print(predict_section)

# 예측 결과를 확인하기 위하여, 데이터프레임에 예측 결과 column 추가
df['predict'] = predict_section
print(df.head(30)) # 정답과 예측 결과가 동일한지 확인하기

# 정확도 확인
score = model.evaluate(x_pad, onehot_y)
print(score[1])

# 예측값 중 최댓값, 두 번째로 높은 값을 predict_section에 저장하기
predict_section = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predict_section.append([most, second])
print(predict_section)

# 예측 결과를 확인하기 위하여, 데이터프레임에 예측 결과 column 추가
df['predict'] = predict_section
print(df[['category', 'predict']].head(30)) # 정답과 예측 결과가 동일한지 확인하기

# 정답률(평균) 확인
df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 1
print(df.OX.mean())