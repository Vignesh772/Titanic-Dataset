import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



train=pd.read_csv("C:\\Users\\Welcome\\Desktop\\GenY\\Task 2 Titanic\\train.csv")
test=pd.read_csv('C:\\Users\\Welcome\\Desktop\\GenY\\Task 2 Titanic\\test.csv')
test_y=pd.read_csv("C:\\Users\\Welcome\\Desktop\\GenY\\Task 2 Titanic\\gender_submission.csv")
yy=test_y.iloc[:,[1]].as_matrix()



full_data=[train,test]

#missing data

#checking number of null data
'''for x in full_data:
    c=x['Cabin'].isnull().sum()
    print("Nulll",c)
    
    
    print (x['Title'].drop_duplicates())'''

##embarked
for x in full_data:
    mode=x['Embarked'].mode()
    print(mode)

    x['Embarked']=x['Embarked'].fillna('S')    


##age
    
for x in full_data:
    avg = x['Age'].mean()
    std = x['Age'].std()
    null_count = x['Age'].isnull().sum()
    null_random_list = np.random.randint(avg - std, avg + std, size=null_count)

    x['Age'][np.isnan(x['Age'])] = null_random_list
    x['Age'] = x['Age'].astype(int)
    
'''for x in full_data:
    x['Age'] = x['Age'].fillna(x['Age'].mean())'''
    
    
#family size
for x in full_data:
    x['Family_Size']=x['Parch'] + x['SibSp'] +1
    
#fare
for x in full_data:
    x['Fare'] = x['Fare'].fillna(train['Fare'].median())
        

#extracting titles
for x in full_data:
    x['Title']=x.Name.apply(lambda y:y.split(' ')[1])
    
for x in full_data:
    x['Title']=x.Title.apply(lambda y:y.strip('.'))
    x['Title']=x.Title.apply(lambda y:y.strip(','))

for x in full_data:
    for i in range(0,len(x)):
        if(x['Title'][i] =='Mlle'):
            x['Title'][i]='Miss'
        elif(x['Title'][i] =='Ms'):
            x['Title'][i]='Miss'
        elif(x['Title'][i] =='Mme'):
            x['Title'][i]='Mrs'
        elif(x['Title'][i] !='Mr' and  x['Title'][i] !='Miss'  and  x['Title'][i] !='Mrs'  and x['Title'][i] !='Master'):
            x['Title'][i]='Rare'
     
#cabin
for x in full_data:
    x['Cabin']=x['Cabin'].fillna('X')
    x['Cabin']=x['Cabin'].apply(lambda y:y[0])
    
#mappimg or converting encoding or grouping features

for x in full_data:
    #gender
    x['Sex']=x['Sex'].map({'male':0,'female':1})
    #cabin
    x['Cabin']=x['Cabin'].map({'X':0 ,'A':1 ,'B':2 ,'C':3 ,'D':4 ,'E':5 ,'F':6 ,'G':7,'T':8 })
    
    #Embarked
    x['Embarked']=x['Embarked'].map({'Q':0 ,'S':1 ,'C':2 })
    
    #titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    x['Title'] = x['Title'].map(title_mapping)
    
    
    
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Parch', 'SibSp']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)

#seperating dependent variable
X=train.iloc[:,[1,2,3,4,5,6,7,8]].values 

y=train.iloc[:,[0]].as_matrix()
#feature scalling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X)
test=sc_x.fit_transform(test)
X=X.T
test=test.T
test=np.vstack( [np.ones((1,418),dtype='int') , test])

X=np.vstack( [np.ones((1,891),dtype='int') , X])


#logistic regression classifier

test_size=np.size(test,1)
iterations=100
alpha =0.3
m=len(y)
J=float(0)
cost=[]
ind=[]
H=np.zeros( (len(y),1) , dtype='float')
Theta=np.zeros((len(X),1) ,dtype='float')

for i in range(0,iterations):
    z=(np.dot(Theta.T ,X)).T
    H=1/(1+np.exp(-z))
    k= y*np.log(H) + (1-y)*np.log(1-H)
    J=-(np.sum(k,0) /m)
    Theta=Theta-((alpha/m)*np.dot(X,(H-y)))
    cost.append(J[0])
    ind.append(i)
   
plt.plot(ind,cost)

z=np.dot(Theta.T,test)
y_pred=1/(1+np.exp(-z))

for i in range(0,test_size):
    if(y_pred[0][i]>0.5):
        y_pred[0][i]=1
    else:
        y_pred[0][i]=0
        
y_pred=y_pred.T  
correct=0
#accuracy
for i in range(0,test_size):
    if(y_pred[i][0]==yy[i][0]):
        correct=correct+1
        
accuracy=(correct/test_size) *100
print(accuracy)
plt.show()
      
        
