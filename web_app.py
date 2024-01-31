import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA # feature reduction technique pca
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Heading of App

st.write("""
# Explore different ML Models and Datasets 
To see which is best between them""")

# write datasets names in one box and make a sidebaar
dataset_name = st.sidebar.selectbox(
    'select dataset',
    ('Iris','Breast cancer','Wine')
)

# write model names in box in sidebar
classifier_name = st.sidebar.selectbox(
    'select Classifier',
    ('KNN','SVM','Random Forest')
)

# define a function to load dataset

def get_dataset(dataset_name):
    data=None # it means  in start there is no data
    if dataset_name == "Iris": # condition if data of iris
        data=datasets.load_iris()
    elif dataset_name == "Wine":
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data 
    y=data.target
    return x, y
# now call a function and equal it to x,y variables
x, y = get_dataset(dataset_name)

# shape of dataset
st.write('Shape of Dataset:', x.shape)
st.write('number of classes:', len(np.unique(y))) # y our target variable

# add parameters of different classifeirs to user input

def add_parameter_ui(classifier_name): # ui means user input
    params=dict() # create an empty dictionary
    if classifier_name=='SVM':
        c = st.sidebar.slider('c',0.01,10.0)
        params['c'] = c # degree of correct classification
    elif classifier_name == 'KNN':
        k = st.sidebar.slider('k',1,15)
        params['k'] = k # number of nearest neighbors
    else:
        max_depth=st.sidebar.slider('max_depth',2,15)
        params['max_depth']=max_depth # depth of every tree grow in random forest
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators # number of trees
    return params
# calling a function
params=add_parameter_ui(classifier_name)

# make classifier based on classifier name and params
def get_classifier(classifier_name, params):
    clf = None  # start with no classifier

    if classifier_name == 'SVM':
        clf = SVC(C=params['c'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'], random_state=1234)
    
    return clf
# Make a checkbox to add piece of code in web app.
if st.checkbox('Show code'):
    with st.echo():
        # Example of using the function
        clf = get_classifier(classifier_name, params)
        # split dataset into train and test data
        x_train, x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=1234)
        # train a classifier
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_test)
        # check accuracy score of model and print on App
        acc = accuracy_score(y_test, y_pred)
# Example of using the function
clf = get_classifier(classifier_name, params)


# split dataset into train and test data
x_train, x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=1234)

# train a classifier
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

# check accuracy score of model and print on App

acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}') # f is string
st.write(f'Accuracy=', acc) # use to change accuracy when model changed

# plot dataset

pca=PCA(2)
x_projected=pca.fit_transform(x)

# ab ham apna data 0 or 1 dimension main slice kr k dain ga
x1=x_projected[:, 0]
x2=x_projected[:, 1]

fig=plt.figure()
plt.scatter(x1,x2,
            c=y,alpha=0.8,
            cmap='viridis')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)











