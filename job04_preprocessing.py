import pickle
import pandas as pd
import numpy as np
from scipy.ndimage import labeled_comprehension
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt, Komoran  # 한국어를 처리하기 위한 라이브러리
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('./crawling_data/naver_titles_total_250416.csv') # 다른 사람이 크롤링한 데이터 파일...
df.info()
print(df.head(30))

print(df.category.value_counts())

X = df.titles
Y = df.category

print(X[0])
# Okt
okt = Okt() # Okt 객체
okt_x = okt.morphs(X[0]) # 형태소 별로 문장 나누기
print(okt_x)
okt_x = okt.morphs(X[0], stem=True) # 단어를 원형으로 변경
print(okt_x)

# Komoran
komoran = Komoran() # Komoran 객체
komoran_x = komoran.morphs(X[0])
print(komoran_x)

print(X[1])
# Okt
okt = Okt() # Okt 객체
# okt_x = okt.morphs(X[1]) # 형태소 별로 문장 나누기
# print(okt_x)
# okt_x = okt.morphs(X[1], stem=True) # 단어를 원형으로 변경
# print(okt_x)

# 형태소에 대하여 라벨링
encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:5])
label = encoder.classes_ # 라벨 부여하기(라벨 순서는 오름차순으로)
print(label)
with open('./models/encoder.pickle', 'wb') as f: # pickle 파일로 저장하기
    pickle.dump(encoder, f)

# One-hot Encoding
onehot_y = to_categorical(labeled_y)
print(onehot_y)

# 한글을 제외한 문자를 공백으로 대체하기
import re
# cleaned_x = re.sub('[^가-힣]', ' ', X[1])
# print(X[1])
# print(cleaned_x)


# X에 있는 문장들을 형태소 단위로 변환
for i in range(len(X)):
    X[i] = re.sub('[^가-힣]', ' ', X[i]) # 모든 문장에 대해 한글만 남기기
    X[i] = okt.morphs(X[i], stem=True) # 형태소 변환
    if i % 1000 == 0:
        print(i)
print(X[:10])

# 한 글자 단위의 형태소 제거하기
for idx, sentence in enumerate(X):
    print(sentence)
    words = []
    for word in sentence:
        print(word)
        if len(word) > 1:
            words.append(word) # 형태소의 길이가 1보다 큰 것만 words에 추가
    print(words[:10])
    X[idx] = ' '.join(words) # 형태소들을 이어 붙인다(사이에 공백을 넣음)
    # print(X[idx]) # X[idx]는 문자열 형태
print(X[:10])

# 토큰화
# 형태소에 라벨 붙이기
token = Tokenizer()
token.fit_on_texts(X)
tokened_x = token.texts_to_sequences(X) # 형태소에 라벨 부여
print(tokened_x[:10])
wordsize = len(token.word_index) + 1 # token의 word_index의 갯수(길이) + '\0' 갯수
print(wordsize)

# 최댓값을 찾는 알고리즘
max = 0
for sentence in tokened_x:
    if max < len(sentence):
        max = len(sentence)
print(max)

with open('./models/token_max_{}.pickle'.format(max), 'wb') as f:
    pickle.dump(token, f)

# 토큰 리스트의 길이를 최댓값으로 통일하기
x_pad = pad_sequences(tokened_x, max)
print(x_pad[:10])

# 데이터셋 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x_pad, onehot_y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np.save('./crawling_data/title_x_train_wordsize{}'.format(wordsize), x_train)
np.save('./crawling_data/title_x_test_wordsize{}'.format(wordsize), x_test)
np.save('./crawling_data/title_y_train_wordsize{}'.format(wordsize), y_train)
np.save('./crawling_data/title_y_test_wordsize{}'.format(wordsize), y_test)