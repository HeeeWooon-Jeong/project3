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
                                                                                                                        # 컴퓨터에 오라클이 두개라서 필요한 코드라 삭제해도됨
app = Flask(__name__)
CORS(app)

host = 'project-db-cgi.smhrd.com'
port = '1524'
password = '0210'
sid = 'xe'
username = 'URLGA'                                                                                                      # 실제 TNS 이름으로 변경

model_file_path = "best_model_test.h5"                                                                                  # 모델 파일 경로

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    data = request.get_json()                                                                                           # get방식 request 키:밸류 형식에서 url데이터를 변수지정
    url = data.get('url')

    if url is None :
        return jsonify({'error_1': 'URL 정보를 전달 받지 못함.'})
    else :
        print('URL 정보를 전달 받음')

    if not os.path.isfile(model_file_path):
        print(f"error_2 :'{model_file_path}' 모델을 찾지 못함. Line-39")
        return jsonify({'error_2': '모델 로드 오류'})
    else:                                                                                                               # 모델이 잘 불러와진다면 IF ELSE 삭제
        if url is not None :                                                                                            # URL이 불러와졌다면 진입
            try :
                print("오라클 연결직전1")
                #dsn = cx_Oracle.makedsn(host, port, sid)                                                                              # DB url 존재 여부 조회 시도
                print("오라클 연결직전2")
                connection = cx_Oracle.connect(username, password, host+":"+port+"/"+sid)                                                 # Oracle 데이터베이스에 연결
                print("오라클 연결직전3")
                cursor = connection.cursor()
                print("오라클 연결 확인")

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
                print(connection + '55')

                if result:                                                                                              # DB에 url 존재시 조회 및 조회수 증가 10/21 수정 조회수 불필요
                    MLCS_STATUS = result['MLCS_STATUS']
                    url_name = result['URL_NAME']
                    return jsonify({'url' : url_name , 'MLCS_STATUS' : MLCS_STATUS})                                    # 기존 DB
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
                        connection = cx_Oracle.connect(username, password, host+":"+port+"/"+sid)
                        cursor = connection.cursor()

                        cursor.execute(sql_query_safe)

                        connection.commit()

                        cursor.close()
                        connection.close()
                    else:                                                                                               # 안전함 외로 판단시 쿼리
                        sql_query_unsafe = f"""
                            INSERT INTO T_URL (URL_NAME, MLCS_STATUS, URL_YN, URL_CNT, URL_DATE)
                            VALUES ('{url}', '{MLCS_STATUS}', 'Y', 1, SYSDATE)
                        """
                        connection = cx_Oracle.connect(username, password, host+":"+port+"/"+sid)
                        cursor = connection.cursor()

                        cursor.execute(sql_query_unsafe)
                        connection.commit()

                        cursor.close()
                        connection.close()

                    return jsonify({'url' : url , 'MLCS_STATUS' : MLCS_STATUS })
            except cx_Oracle.Error as error:
                print(error)                                                                       # 셀렉트 오류시 진입
                return jsonify({'error_3-1': '오라클 셀렉트 문 오류'})
        else :
            return jsonify({'error Line- 138': {'url DB에 없음'}})

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
    # host에 본인 아이피 주소 넣으세요
    #app.run(host='172.30.1.57', port=5000)

    # http://172.30.1.57:5000/ 로 접속
