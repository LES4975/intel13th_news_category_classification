import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.models import *
from keras.layers import *

# 데이터셋 불러 오기
x_train = np.load('./crawling_data/title_x_train_wordsize15396.npy', allow_pickle=True)
x_test = np.load('./crawling_data/title_x_test_wordsize15396.npy', allow_pickle=True)
y_train = np.load('./crawling_data/title_y_train_wordsize15396.npy', allow_pickle=True)
y_test = np.load('./crawling_data/title_y_test_wordsize15396.npy', allow_pickle=True)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 모델 만들기
model = Sequential()
# 15396차원의 의미 공간 생성, 데이터를 의미 유사도에 따라 벡터화하는 레이어를 만든다.
model.add(Embedding(15396, 300)) # 데이터 차원이 너무 많으면 데이터가 희소해진다(차원의 저주). 이를 방지하기 위해 300차원으로 축소한다. (물론 정보 손실을 최소화하는 편으로)
model.build(input_shape=(None, 25))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu')) # 1차원 컨볼루션 레이어로 시퀀스 데이터 처리
model.add(MaxPool1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True)) # LSTM, 이 LSTM의 셀은 128개다.
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True)) # return_sequences=True: 연산하는 동안 나왔던 값들을 모두 저장한다.
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh')) # 보통 마지막 LSTM 레이어에는 return_sequences를 안 쓴다. -> 출력값을 1개만 내놓는다.
model.add(Dropout(0.3))
model.add(Flatten()) # 평활화
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# 모델 학습시키기
fit_hist = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# 모델 검증하기
score = model.evaluate(x_test, y_test, verbose=0)
print('Final test set accuracy', score[1])

# 모델 저장하기
model.save('./models/news_section_classification_model_{}.h5'.format(score[1]))

# 정확도 결과 시각화
plt.plot(fit_hist.history['val_accuracy'], label = 'val_accuracy')
plt.plot(fit_hist.history['accuracy'], label = 'accuracy')
plt.legend()
plt.show()