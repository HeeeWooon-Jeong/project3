from flask import Flask, request, jsonify
from flask_cors import CORS
import cx_Oracle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

## 사용 IDE : Pycham
## 필요 라이브러리 : Flask, cx_Oracle, TensorFlow, NumPy, scikit-learn

                                                                                                                        # 오라클 클라이언트 라이브러리 초기화
cx_Oracle.init_oracle_client(lib_dir=r"C:\oraclexe\app\instantclient_19_20")                                            # 컴퓨터에 오라클이 두개라서 필요한 코드라 삭제해도됨
                                                                                                                        # 주피터노트북에서 실행시 불필요한 것으로 앎
app = Flask(__name__)
CORS(app)

username = 'DCLTEST'
password = '12345'
dsn = 'xe'                                                                                                              # 실제 TNS 이름으로 변경

model_file_path = "best_model_test.h5"                                                                                  # 모델 파일 경로

@app.route('/test', methods=['GET', 'POST'])
def hello_world():
    data = request.get_json()                                                                                 # get방식 request 키:밸류 형식에서 url데이터를 변수지정
    url = data.get('url')
    # url = 'https://www.youtube.com_test2/' # 예시
    print(url)
    data_type = type(data)
    print(data_type)
    if url is None :
        return jsonify({'error_1': 'URL 정보를 전달 받지 못함.'})
    else :
        print('URL 정보를 전달 받음')
    if not os.path.isfile(model_file_path):
        print(f"error_2 :'{model_file_path}' 모델을 찾지 못함. Line-39")
        return jsonify({'error_2': '모델 로드 오류'})
    else:                                                                                                               # 모델이 잘 불러와진다면 IF ELSE 삭제
        if url is not None :                                                                                            # URL이 불러와졌다면 진입
            print('모델 로드완료')
            try :                                                                                                       # DB url 존재 여부 조회 시도
                print('오라클 연결시도')
                connection = cx_Oracle.connect(username, password, dsn)                                                 # Oracle 데이터베이스에 연결
                cursor = connection.cursor()
                # SELECT SQL 쿼리
                sql_query_select = f"""                                                                                         
                    SELECT MLCS_STATUS , URL_NAME 
                    FROM T_URL
                    WHERE URL_NAME = '{url}'
                """
                cursor.execute(sql_query_select)                                                                        # SELECT 쿼리 실행
                result = cursor.fetchone()                                                                              # 결과를 가져오기

                cursor.close()                                                                                          # 커서와 연결 닫기
                connection.close()
                if result:                                                                                              # DB에 url 존재시 조회 및 조회수 증가
                    pre_type = result[0]
                    url_name = result[1]
                    connection = cx_Oracle.connect(username, password, dsn)
                    cursor = connection.cursor()
                    sql_query_cnt = f"""
                        UPDATE T_URL
                        SET URL_CNT = URL_CNT + 1
                        WHERE URL_NAME = '{url}'
                    """
                    try:                                                                                                # URL_CNT 증가를 위한 UPDATE 쿼리 실행
                        cursor.execute(sql_query_cnt)
                        connection.commit()
                    except cx_Oracle.Error as error:                                                                    # 조회수 업데이트 실패시 진입
                        print('Line-72 UPDATE_조회수_오류:', error)
                    finally:
                        cursor.close()
                        connection.close()
                        return jsonify({'url' : url_name , 'MLCS_STATUS' : pre_type})                                   # 기존 DB
                else:                                                                                                   # 신규 URL 일시 진입

                    model = load_model(model_file_path)                                                                 # 모델 로딩

                    tokenizer = Tokenizer(num_words=10000)                                                              # Tokenizer 인스턴스 생성

                    url_sequences = tokenizer.texts_to_sequences([url])                                                 # URL 텍스트를 시퀀스로 변환
                    url_sequences = pad_sequences(url_sequences, maxlen=240)

                    predictions = model.predict(url_sequences)                                                          # 예측

                    labels = ["phishing", "benign", "defacement", "malware"]                                            # 예측된 레이블 출력
                                                                                                                        # 레이블 정의
                    label_encoder = LabelEncoder()                                                                      # LabelEncoder 인스턴스 생성
                    label_encoder.fit(labels)                                                                           # 레이블에 맞게 학습

                    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

                    MLCS_STATUS = predicted_labels[0]
                    if predicted_labels[0] == 'benign' :                                                                # 안전함으로 판단시 쿼리
                        sql_query_safe = f"""
                            INSERT INTO T_URL (URL_NAME, MLCS_STATUS, URL_YN, URL_CNT, URL_DATE)
                            VALUES ('{url}', '{MLCS_STATUS}', 'Y', 1, SYSDATE)
                        """
                        connection = cx_Oracle.connect(username, password, dsn)
                        cursor = connection.cursor()

                        cursor.execute(sql_query_safe)

                        connection.commit()

                        cursor.close()
                        connection.close()
                        print(url, MLCS_STATUS)
                    else:                                                                                               # 안전함 외로 판단시 쿼리
                        sql_query_unsafe = f"""
                            INSERT INTO T_URL (URL_NAME, MLCS_STATUS, URL_YN, URL_CNT, URL_DATE)
                            VALUES ('{url}', '{MLCS_STATUS}', 'Y', 1, SYSDATE)
                        """
                        connection = cx_Oracle.connect(username, password, dsn)
                        cursor = connection.cursor()

                        cursor.execute(sql_query_unsafe)
                        connection.commit()

                        cursor.close()
                        connection.close()
                        print(url, MLCS_STATUS)
                    return jsonify({'url' : url , 'MLCS_STATUS' : MLCS_STATUS })
            except cx_Oracle.Error as error:                                                                            # 셀렉트 오류시 진입
                return jsonify({'error_3-1': '오라클 셀렉트 문 오류'})
        else :
            return jsonify({'error Line- 138': {'url DB에 없음'}})

if __name__ == '__main__':
    #app.run()
    # host에 본인 아이피 주소 넣으세요
    app.run(host='0.0.0.0', port=5000, debug=True)
    # http://172.30.1.57:5000/ 로 접속
