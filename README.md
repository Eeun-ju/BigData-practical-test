# BigData-practical-test
빅데이터분석기사 실기 대비 공부 데이터자격시험 : https://www.dataq.or.kr/www/sub/a_07.do

|주요항목|세부항목|세세항목|
|---------|-------|-------|
|데이터 수집 작업| 데이터 수집하기 | 정형, 반정형,비정형 등 다양한 형태의 데이터를 읽을 수 있다. <br> 필요시 공개 데이터를 수집할 수 있다.|
|데이터 전처리 작업| 데이터 정제하기 | 정제가 필요한 결측값, 이상값 등이 무엇인지 파악할 수 있다. <br> 결측값과 이상값에 대한 처리 기준을 정하고 제거 또는 임의의 값으로 대체할 수 있다.|
|데이터 전처리 작업|데이터 변환하기| 데이터의 유형을 원하는 형태로 변환할 수 있다. <br> 데이터의 범위를 표준화 또는 정규화를 통해 일치시킬 수 있다. <br> 기존 변수를 이용하여 의미 있는 새로운 변수를 생성하거나 변수를 선택할 수 있다.|
|데이터 모형 구축 작업|분석모형 선택하기| 다양한 분석모형을 이해할 수 있다. <br> 주어진 데이터와 분석 목적에 맞는 분석모형을 선택할 수 있다. <br> 선정모형에 필요한 가정 등을 이해할 수 있다.|
|데이터 모형 구축 작업| 분석모형 구축하기| 모형 구축에 부합하는 변수를 지정할 수 있다.<br> 모형 구축에 적합한 형태로 데이터를 조작할 수 있다. <br> 모형 구축에 적절한 매개변수를 지정할 수 있다.|
|데이터 모형 평가 작업| 구축된 모형 평가하기 | 최종 모형을 선정하기 위해 필요한 모형 평가 지표들을 잘 사용할 수 있다. <br> 선택한 평가지표를 이용하고 구축된 여러 모혀을 비교하고 선택할 수 있다. <br> 성능 향상을 위해 구축된 여러 모형을 적절하게 결합할 수 있다. |
|데이터 모형 평가 작업| 분석결과 활용하기| 최종모형 또는 분석결과를 해석할 수 있다. <br> 최종모형 또는 분석결과를 저장할 수 있다.|







## 단답형
+ 정형데이터 수집 방식 및 기술  

|수집 방식|설명|
|------|-----|
|ETL| 수집 대상 데이터를 추출, 가공하여 데이터 웨어하우스 및 데이터 마트에 저장하는 기술|
|FTP| TCP/IP 기반으로 파일을 송수신하는 응용계층 통신 프로토콜|
|API| 솔루션 제조사 및 3rd party 소프트웨어로 제공되는 도구 시스템 간 연동을 통해 실시간으로 데이터를 송수신하는 인터페이스 기술|
|DbtoDB| 데이터베이스 시스템 간 데이터를 동기화하거나 전송하는 기능을 제공하는 기술|
|Rsync| 원격으로 파일과 디렉터리를 동기화하는 응용 프로그램 활용 기술|
|Sqoop| 관계형 데이터베이스와 하둡 간 데이터 전송 기능을 제공하는 기술|

+ 반정형데이터 수집 방식 및 기술

|수집 방식|설명|
|------|-----|
|Sensing| 센서로부터 수집 및 생성된 데이터를 수집하는 기술|
|Streaming|센서 데이터, 미디어 데이터를 실시간으로 수집하는 기술|
|Flume| 로그 데이터를 Event와 Agent를 통해 수집하는 기술|
|Scribe| 로그 데이터를 실시간으로 수집하는 기술|
|Chukwa| Agent와 Collector 구성을 통해 데이터를 수집하고, 하둡에 저장하는 기술 |

+ 비정형데이터 수집 방식 및 기술

|수집 방식|설명|
|------|-----|
|Crawling| 다양한 웹 사이트로부터 데이터를 수집하는 기술|
|RSS| XML 기반으로 정보를 배포하는 프로토콜을 활용하여 데이터를 수집하는 기술|
|Open API| 공개된 API를 이용하여 데이터를 수집하는 기술|
|Scrapy| 파이썬 언어 기반으로 크롤링하여 데이터를 수집하는 기술|
|Apache Kafka| 대용량 실시간 로그 처리를 위한 분산 스트리밍 플랫폼 기술 |



________________________________________
## 간편한 코드


    union_data['주구매지점'].astype("category") # 카테고리 내용 확인 코드
    OneHot_지점 = pd.get_dummies(X_train['주구매지점']) #문자형 -> oneHot 바로 해주는 코드 
    
    
 분류 모델
    
    from sklearn.linear_model import LogisticRegression #Logistic Regression 모듈
    model = LogisticRegression()
    model.fit(train_x train_y)
    
모델 정확도 
       
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score
    
    print("accuracy: %.2f" %accuracy_score(valid_y, pred_y))
    print("Precision : %.3f" % precision_score(valid_y, pred_y))
    print("Recall : %.3f" % recall_score(valid_y, pred_y))
    print("F1 : %.3f" % f1_score(valid_y, pred_y))
    
이상치 탐지
    
    q25, q75 = np.quantile(data,0.25), np.quantile(data,0.75)
    cut_off = (q75- q25)*1.5 #IQR*1.5

    lower = q25 - cut_off 
    upper = q75 + cut_off
