# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import make_union
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import mean_squared_error
import sklearn.model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
#Step_1: Splitting the Data into Train and Test Set
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,accuracy_score, classification_report

class Pipeline:
    "Carry out standard Data Science processes including data preprocessing, model building, "
    "model parameter optimizion, and validating the model"
    
    def __init__(self, data, target=None):
        self.target = target
        self.data = data
        

    def display_data(self, data, describe = True, display_dtype = True):
        "Input: A pandas DataFrame, display data shape, null values, number of duplicated records,"
        "optionally display descriptive statistics of data and info wrt variable dtype"
        
        self.data = data
        display(data.head())
        print('Shape of the Data set:\n', self.data.shape)
        print('\n Total Null Values:\n', self.data.isnull().sum())
        print("\n Total duplicated records:\n", self.data.duplicated().sum())
        if display_dtype:
            print('\n Dtype INFO:\n', self.data.dtypes)
        if describe:
            print("\n Descriptive statistics: \n", self.data.describe())
        
        
    def plot(self,  data, plot_type = 'reg'):
        "Input: Pandas DataFrame, plot pairplot for any number of numerical features"
       
        self.data = data
        sns.pairplot(data, kind = plot_type)
        
        
    def change_col_dtypes(self,data, dtype_col_dict = {}):
        "Input: A pandas DataFrame and a dictonary consisting of col names and dtype they need to be converted into"
        "Return: pandas Dataframe with columns converted to the specified dtypes"
        
        self.data = data
        self.data = self.data.astype(dtype_col_dict)
        
        return self.data
        
        
    def filtering_outliers_using_n_standard_deviation(self, data, num_cols_list = None, n = 2):
        "Input: pandas DataFrame, a list of cols to carryout outlier filtering on, a threshold point in the form of"
        "n standard deviation away from mean,Remove rows that are n std dev away from mean for selected cols"
        "Return: Filtered DataFrame"
        
        self.data = data
        for col in self.data.columns:
            if col in num_cols_list:
                mean = self.data[col].mean()
                std_dev = self.data[col].std()
                upper = mean + abs(n) * std_dev
                lower = mean - abs(n) * std_dev
                bad_idx_list = []
                bad_idx = self.data[(self.data[col] >= upper) | (self.data[col] <= lower)].index
                bad_idx_list.append(bad_idx)
                flat_list = [item for sublist in bad_idx_list for item in sublist]
                clean_data = data.loc[~self.data.index.isin(bad_idx)]

        return clean_data
    
        
    def missing_value_info_and_treatment(self, data, threshold_pct = 100):
        "Input: pandas DataFrame, threshold_pct refers to a threshold perecentage(1-100) of allowing missing values" 
        "to exist in column else drop the column"
        "Return: Filtered data/Original DataFrame"
        
        self.data = data
        print('Inital shape of data', self.data.shape)
        total = self.data.isnull().sum().sort_values(ascending=False)
        percent = (self.data.isnull().sum()/self.data.isnull().count()*100).sort_values(ascending=False)
        dtype = self.data.dtypes
        missing_data = pd.concat([total, percent, dtype], axis=1, keys=['Number of Missing Values', 
                                                                        'Missing Values Percentage', "dtype"])
        missing_data = missing_data.sort_values('Missing Values Percentage',ascending= False)
        display(missing_data)
        good_cols = missing_data[missing_data['Missing Values Percentage'] < threshold_pct].index.tolist() 
        self.data = self.data[good_cols]
        print('Duplicates being Dropped...')
        print('Number of duplicates in the dataset ', self.data.duplicated().sum())
        self.data.drop_duplicates(inplace = True)
        print('shape of data after the treatment', self.data.shape)

        return self.data
        
    
    def fillna_for_numerical_and_categorical_variables(self, data, num_cols = None, cat_cols = None, drop_cols = None):
        "Input: pandas DataFrame,Impute missing value in numerical columns by mean and categorical columns by the mode"
        "for the selected list of num_cols and cat_cols, drop columns by selecting cols as a list using drop_cols"
        "Return: pandas DataFrame with imputation on missing values"
        
        self.data = data
        self.data.drop(columns = drop_cols, inplace = True)
        
        for num_col in num_cols:
            self.data[num_col].fillna(self.data[num_col].mean(), inplace = True)
        
        for cat_col in cat_cols:
            self.data[cat_col].fillna(self.data[cat_col].mode()[0], inplace = True)
        return self.data
    
    
    def encoding_categorical_cols(self, data, encode_cols_list = None, one_hot_encode_list = None):
        "Input: pandas DataFrame, list of columns to LabelEncode and OnehotEncode"
        "Return: Encode columns inplace in the Input DataFrame"
        
        self.data = data
        for col in encode_cols_list:
            label_encode = LabelEncoder()
            self.data[col] = label_encode.fit_transform(self.data[col].astype(str))
            
        for col in one_hot_encode_list:
            self.data[col] = pd.get_dummies(self.data[col])
            
        return self.data
    
    
    def scaling_numerical_features(self, data, scale_cols_list):
        "Input: pandas DataFrame, list of cols to be scaled (standard scaler) using scale_cols_list"
        "Return: pandas DataFrame with scaling done inplace for selected cols"
        
        self.data = data
        scaled_features = self.data.copy()
        features = scaled_features[scale_cols_list]
        scaler = StandardScaler()
        scaled_num_cols = scaler.fit_transform(features.values)
        self.data[scale_cols_list] = scaled_num_cols
        
        return self.data
    
    
    def feature_importance(self, data, thresold_importance = 0):
        "Input: pandas DataFrame, threshold importance(0-1) the min % of variance below which cols will be dropped"
        "Return: Filtered DataFrame"
        
        self.data = data
        model = RandomForestClassifier(100, oob_score = True, random_state=99)
        features = self.data.drop(columns=[self.target])
        labels = self.data[self.target]
        model.fit(features,labels)
        feature_importance = pd.Series(model.feature_importances_, index = features.columns).sort_values()
        feature_importance.plot( kind = 'barh', figsize = (7,6));
        df = pd.DataFrame(feature_importance, columns= ['importance'])
        good_cols = df[df['importance'] > thresold_importance].index.tolist()
        good_cols.append(self.target)
        return self.data[good_cols]

    
    def kfold_cross_val_with_param_grid(self,data, k = 5, model = None, scoring = None, parameter_grid = []):
        "Input: pandas DataFrame, k to specify number of folds (5 by default), estimator we wish to fit, appropriate"
        "scoring parameter for the estimator, parameter grid"
        "Displays baseline accuracy of model and best parameter values"
        
        self.model = model
        self.data = data
        Y = self.data[self.target]    
        X = self.data.drop(columns = [self.target])
        clf = (getattr(sklearn.ensemble, model))()
        kfold = KFold(n_splits= k, random_state=99)
        results = cross_val_score(clf, X, Y, cv=kfold, scoring=scoring, verbose=3)
        print("Baseline Model accuracy: %.2f (%.2f) Accuracy" % (results.mean(), results.std()))
        
        # Create a classifier object with the classifier and parameter candidates
        clf = GridSearchCV(estimator = clf, param_grid=parameter_grid,scoring=scoring, n_jobs=-1, cv = k)

        # Train the classifier on data's feature and target 
        clf.fit(X, Y)
        
        print('Best params:',clf.best_estimator_.n_estimators)
        print('Best max_depth:',clf.best_estimator_.max_depth)
        
        
    def final_model(self, data, model,best_params,scoring=None, k=5):
        "Input: a pandas DataFrame,best params obtained from grid search,estimator as the model, appropriate scoring"
        "criteria for model. Displays models K-Fold accuracy with best params"
        
        features = self.data.drop(columns=[self.target])
        labels = self.data[self.target]
        clf = (getattr(sklearn.ensemble, model))(**best_params)
        kfold = KFold(n_splits= k, random_state=99)
        results = cross_val_score(clf, features, labels, cv=kfold, scoring=scoring, verbose=5)
        
        print("Model with best params, Results: %.2f (%.2f) Accuracy" % (results.mean(), results.std()))
