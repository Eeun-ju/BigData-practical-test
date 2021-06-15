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

________________________________________ 

데이터 변환

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
    
    lower_list = data[data<lower].index.tolist()
	upper_list = data[data>upper].index.tolist()

회귀 평가 지표
 

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
    
    pred = model1.predict(x_test)
    MAE =  mean_absolute_error(y_test, pred)
    MSE = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, pred)



## 예제 

작업형2번

    import pandas as pd
    import numpy as np
    
    x_test = pd.read_csv('data/X_test.csv', index_col = 0)
    x_train = pd.read_csv('data/X_train.csv',index_col = 0)
    y_train = pd.read_csv('data/y_train.csv',index_col = 0)
    
    #데이터 확인
    print(x_train.head())
    print(x_train.columns)
    
    #결측치 확인
    print(x_train.isnull().sum()) #결측치 존재 col 찾기 -> 65% 결측 : feature 삭제하기
    x_train = x_train.drop('환불금액',axis = 1)
    x_test = x_test.drop('환불금액',axis = 1)
    
    #범주형, 수치형 나누기 : 데이터 처리를 빠르게 하기 위한 분리

    check_col = ['주구매상품','주구매지점']
    nume_col = ['총구매액','최대구매액','내점일수','내점당구매건수','주말방문비율','구매주기']
    cate_train = x_train[check_col]
    nume_train = x_train[nume_col]
    
    #이상치 확인 -> 시각화가 불가능 하므로 이상치 개수가 전체의 10%가 넘지 않는 경우만 upper, lower로 대체
    #나머지는 스케일로 변환
    OneHot_train = pd.get_dummies(x_train[check_col])
    OneHot_test = pd.get_dummies(x_test[check_col])

    x_train = pd.concat([x_train,OneHot_train],axis = 1)
    x_train = x_train.drop(check_col,axis = 1)
    x_train = x_train.drop('주구매상품_소형가전',axis = 1)

    x_test = pd.concat([x_test,OneHot_test],axis = 1)
    x_test = x_test.drop(check_col,axis = 1)

    log_list = ['총구매액', '최대구매액', '내점일수', '내점당구매건수','구매주기']

    from sklearn.preprocessing import RobustScaler
    robustScaler = RobustScaler()
    x_train[log_list] = robustScaler.fit_transform(x_train[log_list])
    x_test[log_list] = robustScaler.fit_transform(x_test[log_list])
    #print(nume_train)

    from sklearn import svm
    from sklearn.model_selection import train_test_split


    train_x, valid_x, train_y, valid_y = train_test_split(x_train,y_train,test_size=0.3,shuffle=True,random_state=25)
    #print(train_y.values.ravel().shape)
    model = svm.SVC(kernel = 'rbf', C = 10.0, probability = True)
    model.fit(train_x,train_y.values.ravel())
    #print(model.score(train_x,train_y.values.ravel()))
    #print(model.score(valid_x,valid_y.values.ravel()))


    final_model = svm.SVC(kernel = 'rbf', C = 10.0, probability = True)
    final_model.fit(x_train, y_train.values.ravel())
    pred = final_model.predict_proba(x_test)
    #print(pred)
    result = pd.DataFrame(pred,index=x_test.index)
    result = result.drop(0,axis = 1)
    result.rename(columns={1:'gender'},inplace=True)
    result.index.name = 'custid'
    result.to_csv('수험번호.csv')
     
    
## 예상문제
#### 작업형 1.
+ 이상치, 결측치 관련 문제
+ 범주형 -> 수치형 one-hot 관련 문제

#### 작업형 2.
+ 분류 - 세 가지 이상의 범주로 분류하기
+ 회귀 - 주택 가격 예측, 
