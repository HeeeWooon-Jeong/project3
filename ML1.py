import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 가상의 데이터 프레임 (실제 데이터로 대체해야 함)

data = pd.read_csv('./drive/MyDrive/딥러닝/data/malicious_phish.csv', encoding='ISO-8859-1')

# 데이터 전처리

X = data['url']

y = data['type']

# 레이블 인코딩
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

# tokenizer 사용
tokenizer = Tokenizer(num_words=10000)

tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)  #

X = pad_sequences(X, maxlen=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=100),
    LSTM(64),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련

num_epochs = 1

batch_size = 32

model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

# 12:00 시작   -3시간 소요

# 모델 평가

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# 새로운 URL 주소

new_url = ["https123231://www.navefr.com/231vdsovjsdhv/12391823-91823-9182-398"]  # 여기에 테스트하려는 URL 주소를 추가하세요

# 텍스트 데이터를 시퀀스로 변환

new_url_sequences = tokenizer.texts_to_sequences(new_url)

new_url_sequences = pad_sequences(new_url_sequences, maxlen=100)  # 시퀀스 길이를 모델의 입력 길이와 일치시킵니다.

# 모델을 사용하여 예측

predictions = model.predict(new_url_sequences)

# 예측 결과를 레이블로 디코딩

predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# 예측된 레이블 출력

print("Predicted Labels:", predicted_labels)