import pandas as pd    # 문자형 데이타를 처리 할 때 사용함
import matplotlib.pyplot as plt
import seaborn as sns  # 이미지 보여주는
import numpy as np     # 숫자형 데이타
ctx = 'C:/Users/ezen/PycharmProjects/ezen_0330/titanic/data/'  # 실제 데이타가 있는 경로

train = pd.read_csv(ctx+'train.csv')
test = pd.read_csv(ctx+'test.csv')
# print(train.head())   # head 는 상단의 5개 정도만 보이게
# df = pd.DataFrame(train)
# print(df.columns)
"""

PassengerId 고객아이디
Survived  생존여부  Survival    0 = No, 1 = Yes
Pclass    승선권 Ticket class    1 = 1st, 2 = 2nd, 3 = 3rd
Name 이름
Sex    성별  Sex    
Age    나이  Age in years    
SibSp   동반한 형제자매 , 배우자 수  # of siblings / spouses aboard the Titanic    
Parch   동반한 부모, 자식 수  # of parents / children aboard the Titanic    
Ticket   티켓번호  Ticket number    
Fare    티켓요금  Passenger fare    
Cabin   객실번호   Cabin number    
Embarked  승선한 항구명    Port of Embarkation   
      C = Cherbourg 쉐브로, Q = Queenstown 퀸스타운, S = Southampton 사우스헴튼
  각 항목(피처) 중에서 관련성과 관련이 많은 데이타만 사용해서 분석한다.    
  분석 해서 파생되는 정보중 계속 사용되는 항목은 신규 항목으로 생성하여 넣어 놓는다.
  
 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked' 

"""
# *****************************************
# 생존율  생존 1 은  38.4  사망 2 은 61.6
# *****************************************
"""
f, ax = plt.subplots(1, 2, figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct="%1.1f%%", ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot('Survived', data=train, ax=ax[1])
ax[1].set_title('Survived')
 plt.show()
"""


"""
데이타는 훈련데이타(train.csv), 목적데이터(test.csv) 두가지로 제공됩니다.
목적데이타는 위 항목에서는 Survived 정보가 빠져있습니다.
그것은 답이기 때문입니다.
"""
# ***********************************************
# 성별 남성은 18.9 / 81.1   여성은  74.2 / 25.8
#*************************************************
"""
f, ax = plt.subplots(1, 2, figsize=(18,8))
train['Survived'][train['Sex'] == "male"].value_counts().plot.pie(explode=[0,0.1], autopct="%1.1f%%", ax=ax[0], shadow=True)
train['Survived'][train['Sex'] == "female"].value_counts().plot.pie(explode=[0,0.1], autopct="%1.1f%%", ax=ax[1], shadow=True)
ax[0].set_title('Survived(Male)')
ax[1].set_title('Survived(Female)')

 plt.show()
"""


# ***********************************************
# 승선권 Pclass 의미가 있다
#*************************************************
"""
df_1 = [train['Sex'], train['Survived']]
df_2 = train['Pclass']
df = pd.crosstab(df_1, df_2, margins=True)
# print(df.head())
"""


"""
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
"""
# ***********************************************
# Embarked 승선한항구
# #*************************************************
"""
f, ax = plt.subplots(2, 2, figsize=(10,15))
sns.countplot('Embarked',data=train, ax=ax[0,0])
ax[0,0].set_title('No. of Passengers Boarded ')
sns.countplot('Embarked',hue='Sex', data=train, ax=ax[0,1])
ax[0,1].set_title('Male - Famale for Embarked')
sns.countplot('Embarked',hue='Survived',data=train, ax=ax[1,0])
ax[1,0].set_title('Embarked of Survived ')
sns.countplot('Pclass',data=train, ax=ax[1,1])
ax[1,1].set_title('Embarked vc  Pclass ')

#plt.show()

"""

# 결측치 제거 - 의미 없는 항목은 분석할 때 뺀다

# train.info().sum()
"""
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
"""
# print(train.isnull().sum())
"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
"""

# 버릴것과 데이타가 없지만 넣어야 할항목을 결정 하고 버리지 않을 항목에는 값을 할당한다

def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['survived', 'dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.show()

#bar_chart('Sex')
#bar_chart('Pclass')
#bar_chart('SibSp')
#bar_chart('Parch')
#bar_chart('Embarked')

# Cabin, Ticket  값삭제 => 분석에 관련이 적은 항목을 지운다

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
#print(train.head())
#print(test.head())


# Embarked 값 가공

s_city = train[train["Embarked"]=='S'].shape[0]
c_city = train[train["Embarked"]=='C'].shape[0]
q_city = train[train["Embarked"]=='Q'].shape[0]

#print("S= {}, c= {}, q ={}".format( s_city,c_city,q_city))
#print("S : ", s_city)
#print("S : ", c_city)
#print("S : ", q_city)

# S= 644, c= 168, q =77

# 머신을 실행 할때 문자는 숫자로 변환하여 머신이 이해 용이 하게 한다.
train = train.fillna({"Embarked":"S"})   # Embarked 항목의 null 에 's' 를 넣어 준자
city_mapping = {"S":1, "C":2, "Q":3}
train["Embarked"] = train["Embarked"].map(city_mapping)
test["Embarked"] = test["Embarked"].map(city_mapping)

#print(train.head())
#print(test.head())

# Name 값 가공하기

combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
#print(pd.crosstab(train['Title'], train['Sex']))

"""
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
"""
for dataset in combine:
    dataset['Title'] \
        = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer'],'Rare')
    dataset['Title'] \
        = dataset['Title'].replace(['Countess', 'Lady', 'Col', 'Sir'], 'Royal')
    dataset['Title'] \
        = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] \
        = dataset['Title'].replace('Mme', 'Mrs')

#print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())

"""
    Title  Survived
0  Master  0.575000
1    Miss  0.701087
2      Mr  0.156673
3     Mrs  0.793651
4      Ms  1.000000
5    Rare  0.250000
6   Royal  1.000000

"""
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # fillna
#train.head()


train = train.drop(['Name','PassengerId'], axis = 1)
test = test.drop(['Name','PassengerId'], axis = 1)
combine = [train,test]
#print(train.head())

# 머신을 실행 할때 문자는 숫자로 변환하여 머신이 이해 용이 하게 한다.

sex_mapping = {"male":0, "female":1}
for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map(sex_mapping)


    # Age 가공
train['Age']  = train['Age'] .fillna(-0.5)
test['Age']  = test['Age'] .fillna(-0.5)
bins = [-1,0,5,12,18,24,35,60, np.inf]
labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)

#print(train.head())

age_title_mapping = {0:"Unknown",1:"Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"}
for x in range(len(train['AgeGroup'])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
for x in range(len(test['AgeGroup'])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]
#print(train.head())

age_mapping = {"Baby":1, "Child":2,"Teenager":3,"Student":4, "Young Adult":5,"Adult":6,"Senior":7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis =1)
test = test.drop(['Age'], axis =1)
#print(train.head())


# fare 처리

train['FareBand'] = pd.qcut(train['Fare'], 4 , labels={1,2,3,4})
test['FareBand'] = pd.qcut(test['Fare'], 4 , labels={1,2,3,4})

train = train.drop(['Fare'], axis =1)
test = test.drop(['Fare'], axis =1)
# print(train.head())

#**************************
# 데이터 모델링
#**************************


train_data = train.drop('Survived', axis = 1)
target = train['Survived']
print(train_data.shape)
print(train.shape)
print(train.info)























