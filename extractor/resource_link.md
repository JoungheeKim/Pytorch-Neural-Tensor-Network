# 논문리뷰 - Deep Learning for Event-Driven Stock Prediction
Xiao Ding, [Deep Learning for Event-Driven Stock Prediction](http://ijcai.org/Proceedings/15/Papers/329.pdf), IJCAI2015

### Key Point :
1. Event embedding learned by Neural Tensor Network
2. Long or short decision learned by Deep Prediction Model(a simple derivative of CNN)

### 1. Event embedding learned by Neural Tensor Network
#### Background
1. 단어(bags of word)만으로는 정확한 맥락(context)을 반영하지 못한다. 따라서 (S, V, O) 라는 문장 구조적 특징을 반영해야 한다.
2. 뉴스 또는 문서에 동일한 문장(S, V, O)은 자주 등장하지 않는다. 따라서 문장을 One-Hot Encoding Vector로 하는 것은 비효율적이다. 단어로 이루어진 문장을 효율적인 Vector 형태로 만들어야 한다.

#### Goal
자동화시스템으로 뉴스에서 문장을 추출하여 Vecotr로 변환

#### Process
1. 뉴스기사를 Scrapping
2. Open IE, ZPar, ReVerb를 이용하여 뉴스기사의 문장에서 주어 동사 목적어 추출
    + 에플이 삼성을 제소한다. -> S(애플), V(제소), O(삼성)
3. Word2Vec를 이용하여 단어를 Vector로 변환
    + S(애플), V(제소), O(삼성) -> S(1,0,0,0), V(0,1,1,0), O(0,1,0,0)
4. NTN모델을 이용하여 주어 동사 목적어 Vector를 문장 Vector로 변환
    + S(1,0,0,0), V(0,1,1,0), O(0,1,0,0) -> Sentence(0, 1, 0, 0)

#### Detail about Neural Tensor Network Model(NTN)
![](images/Figure2_convert.png)
![](images/formula_convert.png)

#### Training Process
![](images/training_process.png)

##### 요약
1. INPUT으로 이벤트 집합 E(주어, 동사, 목적어)를 받는다.
2. 뉴스로부터 추출한 단어사전에서 명사를 랜덤으로 가져와 주어를 교체한 더미 이벤트 집합 E'(주어', 동사, 목적어)를 만든다.
3. 이벤트 집합과 더미 이벤트 집합에서 한개씩 가져와 loss function score를 구하고, 업데이트 한다.

#### Personal Thought
1. Training Process가 빈약하다.
    > training socre function에 대한 언급이 없다.
2. 한글을 적용하기 어렵다.
    > 기사에서 구조적 문장(주어, 동사, 목적어)을 외부 라이브러리에 의존하고 있다. 한글뉴스를 논문의 의도와 맞게 주어, 동사, 목적어로 추출해주는 라이브러리를 찾기가 어렵다.(Knolpy에서 제공하는 형태소 분석기, 품사 판별기로 정확히 구조적 문장을 추출할 수 없다.)  
3. 추출한 문장 Vector가 유용하지 않을것 같다.
    > Training Process는 뉴스 기사에서 추출한 Event(문장)과 뉴스 기사로부터 추출되지 않은 Dummy Event를 Vector Space에서 분리하는 역할을 하고 있다. 즉 Event(삼성이 애플을 제소하다.), Event(이건희 회장이 삼성을 경영하다.) 와 Dummy Event(엘지가 애플을 제소하다.), Dummy Event(스티브잡스가 삼성을 경영하다.)의 공간을 분리하는 방식으로 작동한다. 문장의 Vector Space가 유의미하게 분리되었다고 보기 힘들다. 논문의 참고자료인  Richard Socher, [ Reasoning with neural tensor networks for knowledge base completion](https://cs.stanford.edu/~danqi/papers/nips2013.pdf
),  NIPS  pages 926–934, 2013 를 보면 초기 NTN의 의도는 어떠한 명사와 어떠한 동사가 잘 어울리는가를 학습하기 위한 구조적 설계였다.

### 2. Long or short decision learned by Deep Prediction Model(a simple derivative of CNN)
#### Background
1. Event의 영향은 등장한 시기부터 장기간에 걸쳐 영향력을 행사한다. 
2. 장기적, 중장기적, 단기적 Event를 이용하여 현재 주가에 미칠 영향을 확인한다. 
![](images/news_influence.png)

#### Goal
앞서 추출한 Event Vector를 이용하여 가치의 상승, 하락 여부를 추출한다.

#### Detail about CNN model
![](images/Architecture_CNN.png)

##### 요약
1. 장기적, 중장기적, 단기적 Event Vecotr를 나누어 INPUT으로 이용한다.
2. 장기적, 중장기적 Event에만 CNN Process을 이용하여 Event Vector 와 차원이 같은 Vector 각각 1개가 추출된다.
3. 장기적 대표 Vector, 중장기적 대표 Vector, 단기적 Event Vector들을 INPUT으로 Hidden Layer을 한개를 갖고 있는 Deep learning Process 를 지나면 가치의 상승 하락 값이 추출된다.

#### Personal Thought
1. 이 모델을 이용하여 주가를 예측하는 것은 어려워 보인다.
    + 모델의 목표는 일별 주가(가치) 예측이다. 일별 주가데이터는 1년에 365개 미만으로 생성된다. 따라서 20년의 데이터를 수집하더라도 3650개 밖에 되지 않는다. 딥러닝의 경우 다량의 데이터가 확보되지 않은 환경에서 학습할 경우 OverFitting 될수밖에 없다.
2. 이 모델에 넣을 Event를 선별하는 것은 어려운 문제이다.
    + 일별 뉴스생산량을 일정치 않다. 따라서 때대로 적거나 많은 Event를 추출할수 있을 것이다. 하지만 모델의 Input크기는 고정되어있다. 따라서 Event 일부를 선별해야 하는 Process가 필요하다. '좋은' Evnet가 정의하는 것은 또다른 Task이다. 


### 참고자료
1. Richard Socher, [Reasoning with neural tensor networks for knowledge base completion](https://cs.stanford.edu/~danqi/papers/nips2013.pdf
),  NIPS  pages 926–934, 2013
2. Xiao Ding, [Using structured events to predict stock price movement an empirical investigation](https://www.emnlp2014.org/papers/pdf/EMNLP2014148.pdf), EMNLP, pages 1415–1425
3. Yoon Kim, [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181), 2014 EMNLP, pages 1746–1751
4. Xiao Ding, [Deep Learning for Event-Driven Stock Prediction](http://ijcai.org/Proceedings/15/Papers/329.pdf), IJCAI2015
5. [Paper Summary](https://medium.com/@wenchen.li/deep-learning-for-event-driven-stock-prediction-ab783b322f19), Wenchen Li, 
6. [Paper Summary](https://www.hardikp.com/2017/08/18/deep-rnn-summary/), Hardik Patel, 


### Repo available online
1. Neural Tensor network
    - [Reimplementing Neural Tensor Networks for Knowledge Base Completion (KBC) in the TensorFlow framework](https://github.com/dddoss/tensorflow-socher-ntn)
2. Stock Prediction with Deep Learning
    - [Event-Driven-Stock-Prediction-using-Deep-Learning](https://github.com/vedic-partap/Event-Driven-Stock-Prediction-using-Deep-Learning)
    - [Sentiment-Analysis-in-Event-Driven-Stock-Price-Movement-Prediction](https://github.com/WayneDW/Sentiment-Analysis-in-Event-Driven-Stock-Price-Movement-Prediction)
3. Text Classification with CNN
    - [cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch)

    










