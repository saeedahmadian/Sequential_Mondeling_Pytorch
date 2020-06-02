import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sn
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


class ReadData(BaseEstimator,TransformerMixin):
    def __init__(self,data_dir='./Data',file_name='DS_MiniProject_ANON.csv',header=0):
        assert os.path.join(data_dir), print('The directory {} does not exist'.format(data_dir))
        assert file_name.endswith('.csv'), print('The file name {} is not a csv file'.print(file_name))
        self.data_dir = data_dir
        self.file_name = file_name
        self.header = header

    def fit(self,X=None,y=None):
        return self

    def transform(self,X=None,y=None):
        return pd.read_csv('{}/{}'.format(self.data_dir, self.file_name), header=self.header
                               # , index_col='DATE_FOR', parse_dates=True
                           )


class CleanData(BaseEstimator,TransformerMixin):
    def __init__(self,na_method='median'):

        self.na_method= na_method.lower()
        self.list_actions=['mean','median','random','drop','bfill','ffill','pad']
    def fit(self,X=None,y=None):
        return self

    def transform(self,X,y=None):
        self.categorical_features = X.select_dtypes('object').columns.to_list()
        self.numerical_feature = [num for num in X.columns.to_list() if num not in self.categorical_features]
        stats = X.describe()
        if self.na_method not in self.list_actions:
            print('plesae choose one of the followings {}'.format(self.list_actions))
            return None

        if self.na_method in self.list_actions[4:]:
            X.fillna(method=self.na_method,inplace=True)
            return X
        elif self.na_method== 'drop':
            X.dropna(inplace=True)
            return X
        elif self.na_method=='rand':
            randval= random.normalvariate(stats.loc['mean',self.numerical_feature],
                                          stats.loc['std',self.numerical_feature]).to_list()
            args = dict([(col, val) for col, val in zip(self.numerical_feature, randval)])
            X.fillna(value=args,inplace=True)
            X[self.categorical_features].fillna(method='pad',inplace=True)
            return X
        elif self.na_method == 'mean':
            meanval = stats.loc[self.na_method, self.numerical_feature]
            args = dict([(col, val) for col, val in zip(self.numerical_feature, meanval)])
            X.fillna(value=args, inplace=True)
            X[self.categorical_features].fillna(method='pad',inplace=True)
            return X
        elif self.na_method == 'median':
            meanval = stats.loc['50%', self.numerical_feature]
            args = dict([(col, val) for col, val in zip(self.numerical_feature, meanval)])
            X.fillna(value=args, inplace=True)
            X[self.categorical_features].fillna(method='pad',inplace=True)
            return X

class CatToNum(BaseEstimator,TransformerMixin):
    def __init__(self,encoding='ordinal'):
        self.encoding=encoding
    def fit(self,X=None,y=None):
        return self

    def transform(self,X,y=None):
        categorical = X.select_dtypes('object').columns.to_list()
        tmp_df= pd.DataFrame()
        if self.encoding=='one_hot_encoding':
            onehot= OneHotEncoder()
            for col in categorical:
                x_new=onehot.fit_transform(X[col].values.reshape(-1,1)).toarray()
                names = [col + '_{}'.format(i) for i in range(X[col].unique().shape[0])]
                tmp_df[names]= pd.DataFrame(data=x_new,index=X.index,columns=names)
                # X[names]= pd.DataFrame(data=x_new,index=X.index,columns=names)
                X.drop(col,axis=1,inplace=True)
            return pd.concat([tmp_df,X],axis=1)
        else:
            ordenc = OrdinalEncoder()
            tmp= X[categorical].values
            conv = ordenc.fit_transform(tmp)
            X[categorical]= conv
            return X


def Visualize_scatter(DataFrame,label_name='Call_Flag',nrows=9,save_dir='./Figs',name='scatter_plot'):
    n_features= DataFrame.shape[1]-1
    ncols= int(n_features/nrows)
    column_name= DataFrame.columns
    c=0
    for row in range(nrows+1):
        plt.figure(figsize=(18, 30))
        fig, ax = plt.subplots(nrows=1, ncols=ncols, sharex='none', sharey='none')
        for col in range(ncols):
            if c <= n_features:
                neg_samples= DataFrame[column_name[c]][DataFrame[label_name]==0]
                pos_samples= DataFrame[column_name[c]][DataFrame[label_name]==1]
                N_neg=neg_samples.shape[0]
                N_pos= pos_samples.shape[0]
                ax[col].scatter(x=np.arange(0,N_neg),y=neg_samples,
                                     c='blue',alpha=.8,s=40,cmap='viridis',label='{}_"0"_samples'.format(N_neg))
                ax[col].scatter(x=np.arange(0, N_pos), y=pos_samples,
                                c='red', alpha=.8, s=40, cmap='viridis', label='{}_"1"_samples'.format(N_pos))
                ax[col].set_xlabel('sample number')
                ax[col].set_title(column_name[c])
                ax[col].legend(loc='best')
            c+=1
        fig.tight_layout(pad=.4)
        fig.savefig(save_dir + '/{}_{}.png'.format(name, row))
        plt.clf()

    print('Please go to {} to see figures'.format(save_dir))

def Visualize_distribution(DataFrame,label_name='Call_Flag',nrows=9,save_dir='./Figs',name='distribution_plot'):
    n_features= DataFrame.shape[1]-1
    ncols= int(n_features/nrows)
    column_name= DataFrame.columns
    c=0
    for row in range(nrows+1):
        plt.figure(figsize=(18, 30))
        fig, ax = plt.subplots(nrows=1, ncols=ncols, sharex='none', sharey='none')
        for col in range(ncols):
            if c <= n_features:
                neg_samples= DataFrame[column_name[c]][DataFrame[label_name]==0]
                pos_samples= DataFrame[column_name[c]][DataFrame[label_name]==1]
                N_neg=neg_samples.shape[0]
                N_pos= pos_samples.shape[0]
                sn.distplot(neg_samples,
                            bins=80,hist=True,color='blue',ax=ax[col],label='{}_neg_samples'.format(N_neg))
                sn.distplot(pos_samples,
                            bins=80, hist=True, color='orange', ax=ax[col],label='{}_pos_samples'.format(N_pos))

                ax[col].legend(loc='best')
                ax[col].set_title(column_name[c])
            c+=1
        fig.tight_layout(pad=.5)
        fig.savefig(save_dir + '/{}_{}.png'.format(name, row))
        plt.clf()
    print('Please go to {} to see figures'.format(save_dir))


def Visualize_boxplot(DataFrame,label_name='Call_Flag',nrows=9,save_dir='./Figs',name='box_plot'):
    n_features= DataFrame.shape[1]-1
    ncols= int(n_features/nrows)
    column_name= DataFrame.columns
    c=0
    for row in range(nrows+1):
        plt.figure(figsize=(18, 30))
        fig, ax = plt.subplots(nrows=1, ncols=ncols, sharex='none', sharey='none')
        for col in range(ncols):
            if c <= n_features:
                neg_samples= DataFrame[column_name[c]][DataFrame[label_name]==0]
                pos_samples= DataFrame[column_name[c]][DataFrame[label_name]==1]
                print('step {} pos_samples {} neg_samples {}'.format(c,pos_samples.shape[0],neg_samples.shape[0]))
                ax[col].boxplot(neg_samples, positions=[0], notch=True, widths=0.35,
                                 patch_artist=True, boxprops=dict(facecolor="C0"))
                ax[col].boxplot(pos_samples, positions=[1], notch=True, widths=0.35,
                                 patch_artist=True, boxprops=dict(facecolor="C2"))
                ax[col].set_title(column_name[c])
            c+=1
        fig.tight_layout(pad=.5)
        fig.savefig(save_dir + '/{}_{}.png'.format(name, row))
        plt.clf()
    print('Please go to {} to see figures'.format(save_dir))

def plot_paiplot(data,list_features,save_dir='./Figs',fig_name='pairplot'):
    plt.figure(figsize=(20,20))
    g = sn.pairplot(data.iloc[list_features], kind="reg")
    plt.savefig('{}/{}.png'.format(save_dir,fig_name))

class OutlierDetection(BaseEstimator,TransformerMixin):
    def __init__(self,threshold=3,columns=None,name='z_score',max_quantile=.95,min_quantile=.05):
        self.name= name.lower()
        self.threshold= threshold
        self.max_quantile=max_quantile
        self.min_quantile= min_quantile
        self.columns=columns
    def replace_method_whisker(self,X,features):
        for col in features:
            Q1= X[col].quantile(0.25)
            Q3= X[col].quantile(0.75)
            IQR= Q3-Q1
            max_whisker = Q3 + self.threshold*IQR
            min_whisker = Q1 - self.threshold*IQR
            X.loc[(X[col] > max_whisker), col] = max_whisker
            X.loc[(X[col] < min_whisker), col] = min_whisker
        return X

    def replace_method_z_score(self,X,features):
        stats= X[features].describe()
        for col in features:
            max_value = stats.loc['mean',col]+self.threshold*stats.loc['std',col]
            min_value = stats.loc['mean',col]-self.threshold*stats.loc['std',col]
            X.loc[(X[col]>max_value),col]= max_value
            X.loc[(X[col]< min_value),col]= min_value
        return X

    def replace_with_coustom_quantile(self,X,features):
        for col in features:
            max_value = X[col].quantile(self.max_quantile)
            min_value = X[col].quantile(self.min_quantile)
            IQR_new= (max_value-min_value)*self.threshold
            X.loc[(X[col]>max_value),col]=max_value+IQR_new
            X.loc[(X[col] < min_value), col] = min_value + IQR_new
        return X
    def fit(self,X=None,y=None):
        return self

    def transform(self,X,y=None):
        if self.columns== None:
            print("All features are considered for anomaly detection")
            cols = X.columns[0:-1]
            if self.name=='z_score':
                return self.replace_method_z_score(X,cols)
            elif self.name=='whisker':
                return self.replace_method_whisker(X,cols)
            else:
                return self.replace_with_coustom_quantile(X,cols)

        else:
            if self.name=='z_score':
                return self.replace_method_z_score(X,self.columns)
            elif self.name=='whisker':
                return self.replace_method_whisker(X,self.columns)
            else:
                return self.replace_with_coustom_quantile(X,self.columns)













