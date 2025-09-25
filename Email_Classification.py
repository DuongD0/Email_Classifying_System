import pandas as pd
import string
import matplotlib.pyplot as plt
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
import seaborn as sns
from matplotlib import rcParams
import joblib
import argparse
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / 'Datasets'
MODELS_DIR = BASE_DIR / 'models'
TFIDF_VECTORIZER_PATH = MODELS_DIR / 'tfidf_vectorizer.joblib'
SVC_MODEL_PATH = MODELS_DIR / 'svc_classifier.joblib'

def dataframe_from_items(items, columns):
    if not items:
        return pd.DataFrame(columns=columns)
    index = []
    data = []
    for key, value in items:
        index.append(key)
        if isinstance(value, (list, tuple)):
            data.append(list(value))
        else:
            data.append([value])
    df = pd.DataFrame(data, index=index, columns=columns)
    return df

class Data_Preprocessing:
    emails= pd.DataFrame()
    
    def __init__(self):
        print('Object created......Data Preprocessing starts')
        print('----------------------------------------------------------------')
        self.vectorizer = None
        self.count_vectorizer = None
    
    def read_data(self,input_dataset):
        global emails
        
        print('Reading Data from the csv file')
        emails= pd.read_csv(input_dataset, encoding='latin-1')
        
        print('Prints the first 5 rows of the dataframe')
        print(emails.head())
        print('----------------------------------------------------------------')
        
        print('Number of emails in each label')
        print(emails.Label.value_counts())
        print('----------------------------------------------------------------')
        
        print('A copy of the Email content is created')
        text_feat= emails['Email'].copy()
        print('----------------------------------------------------------------')
        
        print('Calling the text_process function to remove punctuation and stopwords.')
        print('----------------------------------------------------------------')
        print('This might take few minutes')
        text_feat= text_feat.apply(self.text_process)
        print('\n')
        print(text_feat.head())
        
        return text_feat

    
    def text_process(self, text):
        #the text is translated by replacing empty string wth empty string and deleting all the characters found in string.punctuation
        text= text.translate(str.maketrans('','',string.punctuation))
        text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
        return " ".join(text)
    
    def stemmer(self, text):
        #stemming of content
        text = text.split()
        words = ""
        for i in text:
                stemmer = SnowballStemmer("english")
                words += (stemmer.stem(i))+" "
        return words
        
    def feature_creation(self, text_feat):
        print('Initialize the TfIdfVectorizer')
        vectorizer= TfidfVectorizer(stop_words='english')
        features = vectorizer.fit_transform(text_feat)
        self.vectorizer = vectorizer
        print('***********Features created successfully*******************')
        print('--------------------------------')
        print('Features: ', features.shape)
        print('\n')
        return features
    
    def featcreation_countvector(self, text_feat):
        print('Initialize the count vector')
        vector_count= CountVectorizer()
        features_count= vector_count.fit_transform(text_feat)
        self.count_vectorizer = vector_count
        print('Features using count vectorizer created successfully')
        print('----------------------------------')
        print('Features_count: ', features_count.shape)
        print('\n')
        return features_count
                
    def split_train_test(self, features):
        global emails
        features_train, features_test, labels_train, labels_test = train_test_split(features, emails['Label'], test_size=0.3, random_state=111)
        print('Features_train: ', features_train.shape)
        print('Features_test: ', features_test.shape)
        print('Labels_train: ', labels_train.shape)
        print('Labels_test: ', labels_test.shape)
        print('\n')
        return features_train, features_test, labels_train, labels_test

class Parameter_tuning:
    
    def __init__(self, features_train, features_test, labels_train, labels_test):
        print('Parameter tuning started')
        print('----------------------------------------------------------------')
        self.selecting_parameters(features_train, features_test, labels_train, labels_test)
        
    def selecting_parameters(self,features_train, features_test, labels_train, labels_test):
        print('Parameter selection for each classifier started')
        print('-----------------------------------------------')
        print('This will take some time to determine the optimal parameters for each classifier.')
        
        #SVM Classifier
        print('-----------Support Vector Machine-------------')
        pred_scores_SVM = []
        krnl = {'rbf' : 'rbf','polynominal' : 'poly', 'sigmoid': 'sigmoid'}
        for k,v in krnl.items():
            for i in np.linspace(0.05, 1, num=20):
                svc = SVC(kernel=v, gamma=i)
                svc.fit(features_train, labels_train)
                pred = svc.predict(features_test)
                pred_scores_SVM.append((k, [i, accuracy_score(labels_test,pred)]))
        
        #converts key-value pair to dataframe    
        df = dataframe_from_items(pred_scores_SVM, ['Gamma', 'Score'])
        df['Score'].plot(kind='line', figsize=(11,6), ylim=(0.8,1.0))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #K-Nearest Neighbour Classifier
        print('---------------K-Nearest Neighbour-------------')
        pred_scores_KNN = []
        for i in range(3,61):
            knc = KNeighborsClassifier(n_neighbors=i)
            knc.fit(features_train, labels_train)
            pred = knc.predict(features_test)
            pred_scores_KNN.append((i, [accuracy_score(labels_test,pred)]))
            
        df = dataframe_from_items(pred_scores_KNN, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #Multinomial Naive Bayes Classifier
        print('-----------------Multinomial Naive Bayes-----------')
        pred_scores_NB = []
        for i in np.linspace(0.001, 0.1, num=20):
            mnb = MultinomialNB(alpha=i)
            mnb.fit(features_train, labels_train)
            pred = mnb.predict(features_test)
            pred_scores_NB.append((i, [accuracy_score(labels_test,pred)]))
        df = dataframe_from_items(pred_scores_NB, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #Decision Tree Classifier
        print('------------------Decision Tree Classifier-------------------')
        pred_scores_DT = []
        for i in range(2,21):
            dtc = DecisionTreeClassifier(min_samples_split=i, random_state=111)
            dtc.fit(features_train, labels_train)
            pred = dtc.predict(features_test)
            pred_scores_DT.append((i, [accuracy_score(labels_test,pred)]))
        df = dataframe_from_items(pred_scores_DT, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #Logistic Regression
        print('-------------Logistic Regression----------------')
        slvr = {'newton-cg' : 'newton-cg', 'lbfgs': 'lbfgs'}
        pred_scores_logistic = []
        for k,v in slvr.items():
            lrc = LogisticRegression(multi_class='multinomial', solver=v, class_weight = 'balanced')
            lrc.fit(features_train, labels_train)
            pred = lrc.predict(features_test)
            pred_scores_logistic.append((k, [accuracy_score(labels_test,pred)]))
        df = dataframe_from_items(pred_scores_logistic, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #Ensemble Classifiers
        print('Ensembles')
        #Random Forest Classifier
        print('------------Random Forest Classifier--------------')
        pred_scores_RF = []
        for i in range(2,36):
            #n_estimators is the number of tress in the forest
            rfc = RandomForestClassifier(n_estimators=i, random_state=111)
            rfc.fit(features_train, labels_train)
            pred = rfc.predict(features_test)
            pred_scores_RF.append((i, [accuracy_score(labels_test,pred)]))
        df = dataframe_from_items(pred_scores_RF, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #AdaBoost Classifier
        print('---------------AdaBoost Classifier-------------')
        pred_scores_abc = []
        for i in range(25,76):
            #n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
            abc = AdaBoostClassifier(n_estimators=i, random_state=111)
            abc.fit(features_train, labels_train)
            pred = abc.predict(features_test)
            pred_scores_abc.append((i, [accuracy_score(labels_test,pred)]))
        df = dataframe_from_items(pred_scores_abc, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #Bagging Classifier
        print('-----------Bagging Classifier-------------')
        pred_scores_bc = []
        for i in range(2,30):
            #n_est: The number of base estimators in the ensemble.
            bc = BaggingClassifier(n_estimators=i, random_state=111)
            bc.fit(features_train, labels_train)
            pred = bc.predict(features_test)
            pred_scores_bc.append((i, [accuracy_score(labels_test,pred)]))
        df = dataframe_from_items(pred_scores_bc, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #ExtraTrees Classifier
        print('------------ExtraTrees Classifier--------------')
        pred_scores_etc = []
        for i in range(2,30):
            #n_estimators: The number of trees in the forest.
            etc = ExtraTreesClassifier(n_estimators=i, random_state=111)
            etc.fit(features_train, labels_train)
            pred = etc.predict(features_test)
            pred_scores_etc.append((i, [accuracy_score(labels_test,pred)]))
        df = dataframe_from_items(pred_scores_etc, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #SVM with Stochastic Gradient Descent Learning
        print('SVM with Stochastic Gradient Descent Learning')
        pred_scores_sgd_svm = []
        for i in range(-6,1):
            sgd_svm = SGDClassifier(loss="hinge", penalty="l2", alpha=10**i, random_state=111)
            sgd_svm.fit(features_train, labels_train)
            pred = sgd_svm.predict(features_test)
            pred_scores_sgd_svm.append((i, [accuracy_score(labels_test,pred)]))
        df = dataframe_from_items(pred_scores_sgd_svm, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('----------------------------------------------------------------')
        print('\n')
        
        #LR with Stochastic Gradient Descent Learning
        print('LR with Stochastic Gradient Descent Learning')
        pred_scores_sgd_LR = []
        for i in range(-6,1):
            sgd_LR = SGDClassifier(loss="log_loss", penalty="l2", alpha=10**i, random_state=111)
            sgd_LR.fit(features_train, labels_train)
            pred = sgd_LR.predict(features_test)
            pred_scores_sgd_LR.append((i, [accuracy_score(labels_test,pred)]))
        df = dataframe_from_items(pred_scores_sgd_LR, ['Score'])
        df.plot(figsize=(11,6))
        
        print(df[df['Score'] == df['Score'].max()])
        print('************Parameter tunning finished successfully*************')
        print('----------------------------------------------------------------')
        print('\n')
        
class Email_Classifictaion:
    clfs={}
    df= pd.DataFrame()
    
    def __init__(self):
        print('---------Email Classifictaion starts------------')
        print('-------------------------------------------------')
        self.best_model_name = None
        self.best_model = None
        self.best_accuracy = 0.0
    
    def email_length(self):
        global emails
        print('Plotting histograms of length of the emails for each label')
        print('----------------------------------------------------------------')
        rcParams.update({'figure.autolayout': False})
        preferred_styles = (
            'seaborn-v0_8-bright',
            'seaborn-bright',
            'seaborn-v0_8',
            'seaborn',
            'default',
        )
        for style_name in preferred_styles:
            try:
                plt.style.use(style_name)
                break
            except OSError:
                continue
        emails.hist(column='Length', by='Label', bins=50, figsize=(11,5))
        return

    #Function created to fit the classifiers       
    def train_classifier(self,clf, feature_train, labels_train):
        clf.fit(feature_train, labels_train)
        
    #Function created to make predictions
    def predict_labels(self,clf, features):
        return (clf.predict(features))
        
    #Function created to create a classifiation report displaying Precision, Recall and F-score
    def class_report(self,pred):
        print(classification_report(labels_test,pred,labels=["FRAUD","SPAM","NORMAL"],target_names=["FRAUD","SPAM","NORMAL"]))
    
    #Function created to get a confusion matrix
    def conf_matrix(self,pred):
        global emails
        label_feat=emails['Label'].copy()
        label_feat= label_feat.unique()
        conf_mat = confusion_matrix(labels_test, pred, labels=["FRAUD","SPAM","NORMAL"])
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=label_feat, yticklabels=label_feat)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
    #Function to get various plots
    def plot(self,df):
        rcParams.update({'figure.autolayout': True})
        df.plot(kind='bar', ylim=(0.85,1.0), figsize=(14,10), align='center', colormap="Accent")
        plt.xticks(np.arange(14), df.index)
        plt.ylabel('Accuracy Score')
        plt.title('Distribution by Classifier')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0.5, 1), loc=0, borderaxespad=0.)
        
    def classification(self, features_train, features_test, labels_train, labels_test):
        global clfs, df
        print('Initializing classifiers')
        print('----------------------------------------------------------------')
        
        svc = SVC(kernel='sigmoid', gamma=1)
        knc = KNeighborsClassifier(n_neighbors=5)
        mnb = MultinomialNB(alpha=0.01)
        dtc = DecisionTreeClassifier(min_samples_split=3, random_state=111)
        lrc = LogisticRegression(multi_class='multinomial', solver='newton-cg', class_weight = 'balanced')
        rfc = RandomForestClassifier(n_estimators=35, random_state=111)
        abc = AdaBoostClassifier(n_estimators=25, random_state=111)
        bc = BaggingClassifier(n_estimators=9, random_state=111)
        etc = ExtraTreesClassifier(n_estimators=29, random_state=111)
        
        #Create a dictionary for the classifiers
        clfs = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc, 'AdaBoost': abc, 'BgC': bc, 'ETC': etc}
        
        print('Training various classifier models and making predictions')
        pred_scores = []
        for k,v in clfs.items():
            self.train_classifier(v, features_train, labels_train)
            pred = self.predict_labels(v,features_test)
            accuracy = accuracy_score(labels_test,pred)
            pred_scores.append((k, [accuracy]))
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model_name = k
                self.best_model = v
            
        df = dataframe_from_items(pred_scores, ['Score'])
        print('Accuracy Scores of classifiers')
        print(df)
        if self.best_model_name is not None:
            print(f'Current best classifier: {self.best_model_name} (accuracy={self.best_accuracy:.4f})')
        print('\n')
        
        svc.fit(features_train, labels_train)
        pred= svc.predict(features_test)
        print('Classifictaion Report of SVC')
        self.class_report(pred)
        print('Get the confusion matrix')
        self.conf_matrix(pred)
        print('----------------------------------------------------------------')
        print('\n')
        
    def Stochastic_Gradient(self, features_train, features_test, labels_train, labels_test):
        global df
        
        print('Building a SVM classifier with Stochastic Gradient Descent learning')
        sgd_svm = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-5, random_state=111)
        sgd_svm.fit(features_train, labels_train)
        pred1= sgd_svm.predict(features_test)
        k1='SGD_SVM'
        print('Accuracy_score: ',accuracy_score(labels_test,pred1))#0.98555
        print('Classifictaion Report')
        self.class_report(pred1)
        print('Get the confusion matrix')
        self.conf_matrix(pred1)
        print('----------------------------------------------------------------')
        
        print('Building a Logistic Regression model with Stochastic Gradient Descent learning')
        sgd_LR = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=111)
        sgd_LR.fit(features_train, labels_train)
        pred2= sgd_LR.predict(features_test)
        k2='SGD_LR'
        print('Accuracy_score: ',accuracy_score(labels_test,pred2))#0.98222
        print('Classifictaion Report')
        self.class_report(pred2)
        print('Get the confusion matrix')
        self.conf_matrix(pred2)
        print('----------------------------------------------------------------')
        
        pred_SGD=[]
        pred_SGD.append((k1,[accuracy_score(labels_test,pred1)]))
        pred_SGD.append((k2,[accuracy_score(labels_test,pred2)]))
        df2 = dataframe_from_items(pred_SGD, ['Score'])
        df = pd.concat([df, df2])
        print('Accuracy Scores')
        print(df)
        print('----------------------------------------------------------------')
        print('\n')
        
    def Vote(self, features_train, features_test, labels_train, labels_test):
        global df
        
        print('Using Vote')
        svc = SVC(kernel='sigmoid', gamma=1)
        rfc = RandomForestClassifier(n_estimators=35, random_state=111)
        abc = AdaBoostClassifier(n_estimators=25, random_state=111)
        bc = BaggingClassifier(n_estimators=9, random_state=111)
        etc = ExtraTreesClassifier(n_estimators=29, random_state=111)
        
        print('Vote on BGC, ETC, RF and AB with voting="soft"')
        eclf = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc),('AB',abc)], voting='soft')
        eclf.fit(features_train,labels_train)
        pred = eclf.predict(features_test)
        k1= 'Vote(BC,ETC,RF,AB)'
        print('Accuracy_score: ',accuracy_score(labels_test,pred))#0.975
        print('----------------------------------------------------------------')
        
        print('Vote on BGC, ETC and RF with voting="hard"')
        eclf_2 = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc)], voting='hard')
        eclf_2.fit(features_train,labels_train)
        pred_2 = eclf_2.predict(features_test)
        k2= 'Vote(BC,ETC,RF)'
        print('Accuracy_score: ',accuracy_score(labels_test,pred_2))#0.982
        print('----------------------------------------------------------------')
        
        print('Vote on BGC, RF and SVC with voting="hard"')
        eclf_3 = VotingClassifier(estimators=[('BgC', bc), ('RF', rfc),('SVC', svc)], voting='hard')
        eclf_3.fit(features_train,labels_train)
        pred_3= eclf_3.predict(features_test)
        k3='Vote(SVC,BGC,RF)'
        print('Accuracy_score: ',accuracy_score(labels_test,pred_3))#0.985
        print('----------------------------------------------------------------')
        
        pred_vote=[]
        pred_vote.append((k1, [accuracy_score(labels_test,pred)]))
        pred_vote.append((k2, [accuracy_score(labels_test,pred_2)]))
        pred_vote.append((k3, [accuracy_score(labels_test,pred_3)]))
        
        df3 = dataframe_from_items(pred_vote, ['Score'])
        df = pd.concat([df, df3])
        print('Accuracy Scores')
        print(df)
        self.plot(df)
        print('Classifictaion report of Vote(SVC,BGC,RF(hard))')
        self.class_report(pred_3)
        print('Get cofusion matrix of Vote(SVC,BGC,RF(hard))')
        self.conf_matrix(pred_3)
        print('Classifictaion completed successfully')
        print('-------------------------------------')
        print('\n')
        
class Email_Classification_Stemming:
    df4= pd.DataFrame()
    clfs_2={}
    
    def __init__(self):
        print('---------Email Classifictaion with stemming starts------------')
        print('-------------------------------------------------')
        
    def classification_stem(self, features_train, features_test, labels_train, labels_test):
        global clfs_2, df4
        email_class_obj= Email_Classifictaion()
        
        print('Initializing classifiers for stemming with different parameters')
        print('----------------------------------------------------------------')
        svc = SVC(kernel='sigmoid', gamma=0.7)
        knc = KNeighborsClassifier(n_neighbors=5)
        mnb = MultinomialNB(alpha=0.006)
        dtc = DecisionTreeClassifier(min_samples_split=5, random_state=111)
        lrc = LogisticRegression(multi_class='multinomial', solver='newton-cg', class_weight = 'balanced')
        rfc = RandomForestClassifier(n_estimators=22, random_state=111)
        abc = AdaBoostClassifier(n_estimators=65, random_state=111)
        bc = BaggingClassifier(n_estimators=27, random_state=111)
        etc = ExtraTreesClassifier(n_estimators=25, random_state=111)
        
        #Create a dictionary for the classifiers
        clfs_2 = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc, 'AdaBoost': abc, 'BgC': bc, 'ETC': etc}
        
        pred_scores = []
        
        for k,v in clfs_2.items():
            email_class_obj.train_classifier(v, features_train, labels_train)
            pred = email_class_obj.predict_labels(v,features_test)
            pred_scores.append((k, [accuracy_score(labels_test,pred)]))
            
        df4 = dataframe_from_items(pred_scores, ['Score2'])
        print('Accuracy Scores of classifiers')
        print(df4)
        print('----------------------------------------------------------------')
        print('\n')
        
    def Stochastic_Gradient_stem(self, features_train, features_test, labels_train, labels_test):
        global df4
        
        print('Building a SVM classifier with Stochastic Gradient Descent learning')
        sgd_svm = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, random_state=111)
        sgd_svm.fit(features_train, labels_train)
        pred1= sgd_svm.predict(features_test)
        k1='SGD_SVM'
        print('Accuracy_score: ',accuracy_score(labels_test,pred1))#0.98555
        print('----------------------------------------------------------------')
        
        
        print('Building a Logistic Regression model with Stochastic Gradient Descent learning')
        sgd_LR = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=111)
        sgd_LR.fit(features_train, labels_train)
        pred2= sgd_LR.predict(features_test)
        k2='SGD_LR'
        print('Accuracy_score: ',accuracy_score(labels_test,pred2))#0.98222
        print('----------------------------------------------------------------')
        
        pred_SGD=[]
        pred_SGD.append((k1,[accuracy_score(labels_test,pred1)]))
        pred_SGD.append((k2,[accuracy_score(labels_test,pred2)]))
        df5 = dataframe_from_items(pred_SGD, ['Score2'])
        df4 = pd.concat([df4, df5])
        print('Accuracy Scores')
        print(df4)
        print('----------------------------------------------------------------')
        print('\n')
        
    def Vote_Stem(self, features_train, features_test, labels_train, labels_test):
        global df, df4
        email_class_obj= Email_Classifictaion()
        
        print('Using Vote')
        svc = SVC(kernel='sigmoid', gamma=0.7)
        rfc = RandomForestClassifier(n_estimators=22, random_state=111)
        abc = AdaBoostClassifier(n_estimators=65, random_state=111)
        bc = BaggingClassifier(n_estimators=27, random_state=111)
        etc = ExtraTreesClassifier(n_estimators=25, random_state=111)
        
        print('Vote on BGC, ETC, RF and AB with voting="soft"')
        eclf = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc),('AB',abc)], voting='soft')
        eclf.fit(features_train,labels_train)
        pred = eclf.predict(features_test)
        k1= 'Vote(BC,ETC,RF,AB)'
        print('Accuracy_score: ',accuracy_score(labels_test,pred))#0.975
        print('----------------------------------------------------------------')
        
        print('Vote on BGC, ETC and RF with voting="hard"')
        eclf_2 = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc)], voting='hard')
        eclf_2.fit(features_train,labels_train)
        pred_2 = eclf_2.predict(features_test)
        k2= 'Vote(BC,ETC,RF)'
        print('Accuracy_score: ',accuracy_score(labels_test,pred_2))#0.982
        print('----------------------------------------------------------------')
        
        print('Vote on BGC, RF and SVC with voting="hard"')
        eclf_3 = VotingClassifier(estimators=[('BgC', bc), ('RF', rfc),('SVC', svc)], voting='hard')
        eclf_3.fit(features_train,labels_train)
        pred_3= eclf_3.predict(features_test)
        k3='Vote(SVC,BGC,RF)'
        print('Accuracy_score: ',accuracy_score(labels_test,pred_3))#0.985
        print('----------------------------------------------------------------')
        
        pred_vote=[]
        pred_vote.append((k1, [accuracy_score(labels_test,pred)]))
        pred_vote.append((k2, [accuracy_score(labels_test,pred_2)]))
        pred_vote.append((k3, [accuracy_score(labels_test,pred_3)]))
        
        df6 = dataframe_from_items(pred_vote, ['Score2'])
        df4 = pd.concat([df4, df6])
        print(df4)
        df = pd.concat([df,df4],axis=1)
        print('Accuracy Scores')
        print(df)
        email_class_obj.plot(df)
        print('Classifictaion completed successfully')
        print('-------------------------------------')
        print('\n')
        
class Length_Matrix:
    df7= pd.DataFrame()
    
    def __init__(self):
        print('Appending Length feature to the matrix')
    
    def Length_without_stemming(self, features):
        global emails, clfs, df
        Data_Preprocess_obj= Data_Preprocessing()
        email_class_obj= Email_Classifictaion()
        
        print('Without Stemming')
        lf = emails['Length'].to_numpy()
        if hasattr(features, "toarray"):
            features_dense = features.toarray()
        else:
            features_dense = np.asarray(features)
        newfeat = np.hstack((features_dense, lf[:, None]))
        
        print('Splitting features into train and test set')
        features_train, features_test, labels_train, labels_test = Data_Preprocess_obj.split_train_test(newfeat)
        
        print('Training various classifiers')
        pred_scores = []
        for k,v in clfs.items():
            email_class_obj.train_classifier(v, features_train, labels_train)
            pred = email_class_obj.predict_labels(v,features_test)
            pred_scores.append((k, [accuracy_score(labels_test,pred)]))
            
        df7 = dataframe_from_items(pred_scores, ['Score3'])
        print(df7)
        
        print('Building a SVM classifier with Stochastic Gradient Descent learning')
        sgd_svm = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-5, random_state=111)
        sgd_svm.fit(features_train, labels_train)
        pred1= sgd_svm.predict(features_test)
        k1='SGD_SVM'
        print('----------------------------------------------------------------')
         
        print('Building a Logistic Regression model with Stochastic Gradient Descent learning')
        sgd_LR = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=111)
        sgd_LR.fit(features_train, labels_train)
        pred2= sgd_LR.predict(features_test)
        k2='SGD_LR'
        print('----------------------------------------------------------------')
        
        pred_SGD=[]
        pred_SGD.append((k1,[accuracy_score(labels_test,pred1)]))
        pred_SGD.append((k2,[accuracy_score(labels_test,pred2)]))
        df8 = dataframe_from_items(pred_SGD, ['Score3'])
        df7 = pd.concat([df7, df8])
        print('Accuracy Scores')
        print(df7)
        print('----------------------------------------------------------------')
        
        print('Using Vote')
        svc = SVC(kernel='sigmoid', gamma=1)
        rfc = RandomForestClassifier(n_estimators=35, random_state=111)
        abc = AdaBoostClassifier(n_estimators=25, random_state=111)
        bc = BaggingClassifier(n_estimators=9, random_state=111)
        etc = ExtraTreesClassifier(n_estimators=29, random_state=111)
        
        print('Vote on BGC, ETC, RF and AB with voting="soft"')
        vote_l1 = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc), ('Ada', abc)], voting='soft')
        vote_l1.fit(features_train,labels_train)
        pred = vote_l1.predict(features_test)
        k1= 'Vote(BC,ETC,RF,AB)'
        
        print('Vote on BGC, ETC and RF with voting="hard"')
        vote_l2 = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc)], voting='hard')
        vote_l2.fit(features_train,labels_train)
        pred_2 = vote_l2.predict(features_test)
        k2= 'Vote(BC,ETC,RF)'
        
        print('Vote on BGC, RF and SVC with voting="hard"')
        vote_l3 = VotingClassifier(estimators=[('BgC', bc), ('RF', rfc),('SVC', svc)], voting='hard')
        vote_l3.fit(features_train,labels_train)
        pred_3 = vote_l3.predict(features_test)
        k3='Vote(SVC,BGC,RF)'
        
        pred_vote_l1=[]
        pred_vote_l1.append((k1, [accuracy_score(labels_test,pred)]))
        pred_vote_l1.append((k2, [accuracy_score(labels_test,pred_2)]))
        pred_vote_l1.append((k3, [accuracy_score(labels_test,pred_3)]))
        
        df9 = dataframe_from_items(pred_vote_l1, ['Score3'])
        df7 = pd.concat([df7, df9])
        print(df7)
        df = pd.concat([df,df7],axis=1)
        print('Accuracy Scores')
        print(df)
        print('\n')
        email_class_obj.plot(df)
        
    def Length_stemming(self, features_stem):
        global emails, clfs_2, df
        Data_Preprocess_obj= Data_Preprocessing()
        email_class_obj= Email_Classifictaion()
        
        print('With Stemming')
        lf = emails['Length'].to_numpy()
        if hasattr(features_stem, "toarray"):
            features_stem_dense = features_stem.toarray()
        else:
            features_stem_dense = np.asarray(features_stem)
        newfeat_stem = np.hstack((features_stem_dense, lf[:, None]))
        
        print('Split the features into train and test set')
        features_train, features_test, labels_train, labels_test = Data_Preprocess_obj.split_train_test(newfeat_stem)
        
        print('Training various classifiers')
        pred_scores = []
        for k,v in clfs_2.items():
            email_class_obj.train_classifier(v, features_train, labels_train)
            pred = email_class_obj.predict_labels(v,features_test)
            pred_scores.append((k, [accuracy_score(labels_test,pred)]))
            
        df10 = dataframe_from_items(pred_scores, ['Score4'])
        print(df10)
        
        print('Building a SVM classifier with Stochastic Gradient Descent learning')
        sgd_svm = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, random_state=111)
        sgd_svm.fit(features_train, labels_train)
        pred1= sgd_svm.predict(features_test)
        k1='SGD_SVM'
        print('----------------------------------------------------------------')
         
        print('Building a Logistic Regression model with Stochastic Gradient Descent learning')
        sgd_LR = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=111)
        sgd_LR.fit(features_train, labels_train)
        pred2= sgd_LR.predict(features_test)
        k2='SGD_LR'
        print('----------------------------------------------------------------')
        
        pred_SGD=[]
        pred_SGD.append((k1,[accuracy_score(labels_test,pred1)]))
        pred_SGD.append((k2,[accuracy_score(labels_test,pred2)]))
        df11 = dataframe_from_items(pred_SGD, ['Score4'])
        df10 = pd.concat([df10, df11])
        print('Accuracy Scores')
        print(df10)
        print('----------------------------------------------------------------')
        print('\n')
        
        print('Using Vote')
        svc = SVC(kernel='sigmoid', gamma=0.7)
        rfc = RandomForestClassifier(n_estimators=22, random_state=111)
        abc = AdaBoostClassifier(n_estimators=65, random_state=111)
        bc = BaggingClassifier(n_estimators=27, random_state=111)
        etc = ExtraTreesClassifier(n_estimators=25, random_state=111)
        
        print('Vote on BGC, ETC, RF and AB with voting="soft"')
        vote_l1 = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc), ('Ada', abc)], voting='soft')
        vote_l1.fit(features_train,labels_train)
        pred = vote_l1.predict(features_test)
        k1= 'Vote(BC,ETC,RF,AB)'
        
        print('Vote on BGC, ETC and RF with voting="hard"')
        vote_l2 = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc)], voting='hard')
        vote_l2.fit(features_train,labels_train)
        pred_2 = vote_l2.predict(features_test)
        k2= 'Vote(BC,ETC,RF)'
        
        print('Vote on BGC, RF and SVC with voting="hard"')
        vote_l3 = VotingClassifier(estimators=[('BgC', bc), ('RF', rfc),('SVC', svc)], voting='hard')
        vote_l3.fit(features_train,labels_train)
        pred_3 = vote_l3.predict(features_test)
        k3='Vote(SVC,BGC,RF)'
        
        pred_vote_l2=[]
        pred_vote_l2.append((k1, [accuracy_score(labels_test,pred)]))
        pred_vote_l2.append((k2, [accuracy_score(labels_test,pred_2)]))
        pred_vote_l2.append((k3, [accuracy_score(labels_test,pred_3)]))
        
        df12 = dataframe_from_items(pred_vote_l2, ['Score4'])
        df10 = pd.concat([df10, df12])
        print(df10)
        df = pd.concat([df,df10],axis=1)
        print('Accuracy Scores')
        print(df)
        print('\n')
        email_class_obj.plot(df)
        

class RealTimeEmailDetector:
    def __init__(self, vectorizer=None, classifier=None, vectorizer_path=TFIDF_VECTORIZER_PATH, model_path=SVC_MODEL_PATH):
        if vectorizer is None or classifier is None:
            vectorizer, classifier = load_saved_artifacts(vectorizer_path, model_path)
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.stop_words = set(stopwords.words('english'))
    
    def _preprocess(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = [word for word in text.split() if word.lower() not in self.stop_words]
        return " ".join(tokens)
    
    def predict(self, raw_email_text):
        processed = self._preprocess(raw_email_text)
        features = self.vectorizer.transform([processed])
        return self.classifier.predict(features)[0]


def load_saved_artifacts(vectorizer_path=TFIDF_VECTORIZER_PATH, model_path=SVC_MODEL_PATH):
    if not vectorizer_path.exists() or not model_path.exists():
        raise FileNotFoundError('Saved artifacts not found. Run training first or supply the correct paths.')
    print(f'Loading vectorizer from {vectorizer_path}')
    vectorizer = joblib.load(vectorizer_path)
    print(f'Loading classifier from {model_path}')
    classifier = joblib.load(model_path)
    return vectorizer, classifier


def run_training_pipeline():
    global labels_test
    input_dataset = DATASETS_DIR / 'final_dataset.csv'
    if not input_dataset.exists():
        raise FileNotFoundError(f'Dataset not found at {input_dataset}. Please generate the dataset or update the path.')

    Processed_dataset= Data_Preprocessing()
    text_feat= Processed_dataset.read_data(input_dataset)


    '''************************************************************************
    *********************BLOCK-1: WITHOUT STEMMING*****************************
    ***************************************************************************'''

    print('Calling the feature_creation function (TF-TDF)')
    print('----------------------------------------------------------------')
    features= Processed_dataset.feature_creation(text_feat)

    print('Splitting the features into train and test set')
    print('----------------------------------------------------------------')
    features_train, features_test, labels_train, labels_test= Processed_dataset.split_train_test(features)

    param_tuning=Parameter_tuning(features_train, features_test, labels_train, labels_test)

    email_class= Email_Classifictaion()
    email_class.email_length()
    email_class.classification(features_train, features_test, labels_train, labels_test)
    email_class.Stochastic_Gradient(features_train, features_test, labels_train, labels_test)
    email_class.Vote(features_train, features_test, labels_train, labels_test)

    tfidf_vectorizer_no_stem = Processed_dataset.vectorizer
    best_classifier = email_class.best_model
    best_model_name = email_class.best_model_name
    if tfidf_vectorizer_no_stem is not None and best_classifier is not None:
        joblib.dump(tfidf_vectorizer_no_stem, TFIDF_VECTORIZER_PATH)
        joblib.dump(best_classifier, SVC_MODEL_PATH)
        print(f'Saved TF-IDF vectorizer to {TFIDF_VECTORIZER_PATH}')
        print(f'Saved {best_model_name} classifier to {SVC_MODEL_PATH}')
    else:
        print('Warning: Unable to persist artifacts because either the vectorizer or best classifier is missing.')

    '''*************************END BLOCK-1 ************************************'''


    '''********************************************************************
    **************** BLOCK-2: WITH STEMMING *******************************
    ********************************************************************'''
    print('Stemming of the content starts.....')
    text_feat_stem= text_feat.apply(Processed_dataset.stemmer)
    print('----------------------------------------------------------------')

    print('Calling the feature_creation function (TF-TDF)')
    features_stem= Processed_dataset.feature_creation(text_feat_stem)
    print('----------------------------------------------------------------')

    print('Splitting the features into train and test set')
    features_train_stem, features_test_stem, labels_train_stem, labels_test_stem= Processed_dataset.split_train_test(features_stem)
    print('----------------------------------------------------------------')

    param_tuning=Parameter_tuning(features_train_stem, features_test_stem, labels_train_stem, labels_test_stem)

    email_class_stem= Email_Classification_Stemming()
    email_class_stem.classification_stem(features_train_stem, features_test_stem, labels_train_stem, labels_test_stem)
    email_class_stem.Stochastic_Gradient_stem(features_train_stem, features_test_stem, labels_train_stem, labels_test_stem)
    email_class_stem.Vote_Stem(features_train_stem, features_test_stem, labels_train_stem, labels_test_stem)

    '''**************************END BLOCK-2 **************************************'''


    '''********************************************************************************
    *****************************BLOCK-3: LENGTH MATRIX *******************************
    ***********************************************************************************'''

    length_mat= Length_Matrix()
    length_mat.Length_without_stemming(features)
    length_mat.Length_stemming(features_stem)

    '''******************** END BLOCK-3 ***********************************************'''


    '''***********************************************************************************
    ********************* BLOCK-4: COUNT VECTOR ******************************************
    ***************************************************************************************'''

    print('Calling the function for feature cration using Count Vector')
    print('----------------------------------------------------------------')
    features_count= Processed_dataset.featcreation_countvector(text_feat)

    print('Splitting the features into train and test set')
    print('----------------------------------------------------------------')
    features_train_count, features_test_count, labels_train_count, labels_test_count= Processed_dataset.split_train_test(features_count)

    email_class.classification(features_train_count, features_test_count, labels_train_count, labels_test_count)
    email_class.Stochastic_Gradient(features_train_count, features_test_count, labels_train_count, labels_test_count)
    email_class.Vote(features_train_count, features_test_count, labels_train_count, labels_test_count)

    print('Calling the function for feature cration using Count Vector with stemming')
    features_count_stem= Processed_dataset.featcreation_countvector(text_feat_stem)
    print('----------------------------------------------------------------')

    print('Splitting the features into train and test set')
    features_train_count_stem, features_test_count_stem, labels_train_count_stem, labels_test_count_stem= Processed_dataset.split_train_test(features_count_stem)
    print('----------------------------------------------------------------')

    email_class_stem.classification_stem(features_train_count_stem, features_test_count_stem, labels_train_count_stem, labels_test_count_stem)
    email_class_stem.Stochastic_Gradient_stem(features_train_count_stem, features_test_count_stem, labels_train_count_stem, labels_test_count_stem)
    email_class_stem.Vote_Stem(features_train_count_stem, features_test_count_stem, labels_train_count_stem, labels_test_count_stem)

    '''*************************** END BLOCK-4 ***********************************************'''

    return tfidf_vectorizer_no_stem, best_classifier, best_model_name


def main(argv=None):
    parser = argparse.ArgumentParser(description='Email classification training and inference pipeline.')
    parser.add_argument('--skip-training', action='store_true', help='Load saved artifacts for inference without running the training pipeline.')
    parser.add_argument('--force-retrain', action='store_true', help='Suppress warnings about overwriting existing artifacts during retraining.')
    args = parser.parse_args(argv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_exist = TFIDF_VECTORIZER_PATH.exists() and SVC_MODEL_PATH.exists()

    if args.skip_training:
        detector = RealTimeEmailDetector()
        print('Artifacts loaded. RealTimeEmailDetector is ready for live predictions.')
        return detector

    if artifacts_exist:
        if args.force_retrain:
            print('Existing artifacts found. Force retrain enabled; overwriting cached models.')
        else:
            print('Existing artifacts found; the new training run will overwrite them.')
            print('Use --skip-training to reuse cached models or pass --force-retrain to silence this warning.')

    vectorizer, classifier, best_name = run_training_pipeline()
    detector = RealTimeEmailDetector(vectorizer=vectorizer, classifier=classifier)
    if best_name:
        print(f'Best classifier this run: {best_name}')
    print(f'Saved artifacts available in {MODELS_DIR}. Use RealTimeEmailDetector for real-time scoring.')
    return detector


if __name__ == '__main__':
    main()
