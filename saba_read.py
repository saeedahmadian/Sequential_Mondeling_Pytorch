
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler


MAX_VALUE=1
MIN_VALUE=0

dense_features1= ['Age','BMI','CRT0ALC','Total_blood_volume_litres_Nadlerformula']
dense_features2= ['PTV', 'bodyV5_rel','bodyV10_rel','bodyV15_rel',
                'bodyV20_rel','bodyV25_rel','bodyV30_rel','bodyV35_rel','bodyV40_rel',
                'bodyV45_rel','bodyV50_rel','meanbodydose','bodyvolume','lungV5_rel',
                'lungV10_rel','lungV15_rel','lungV20_rel','lungV25_rel','lungV30_rel',
                'lungV35_rel','lungV40_rel','lungV45_rel','lungV50_rel','meanlungdose',
                'lungvolume','heartV5_rel','heartV10_rel','heartV15_rel','heartV20_rel',
                'heartV25_rel','heartV30_rel','heartV35_rel','heartV40_rel','heartV45_rel',
                'heartV50_rel','meanheartdose','heartvolume','spleenV5_rel','spleenV10_rel',
                'spleenV15_rel','spleenV20_rel','spleenV25_rel','spleenV30_rel','spleenV35_rel',
                'spleenV40_rel','spleenV45_rel','spleenV50_rel','meanspleendose','spleenvolume'
                  ]
dense_features= dense_features1+dense_features2
sparse_features = ['IMRT1Protons0','Sex','Race','Histology',
                   'Location_uppmid_vs_low','Location_upp_vs_mid_vs_low','Induction_chemo',
                   'CChemotherapy_type']
sequential_features_t0 = [
    'CRT0neutrophil_percent','CRT0lymphocyte_percent','CRT0monocyte_percent'
    ]
sequential_features_t1=['CRT1lymphocyte_absolute_count_KuL','CRT1lymphocyte_percent']
sequential_features_t2=['CRT2lymphocyte_absolute_count_KuL','CRT2lymphocyte_percent']
sequential_features_t3=['CRT3lymphocyte_absolute_count_KuL','CRT3lymphocyte_percent']
sequential_features_t4=['CRT4lymphocyte_absolute_count_KuL','CRT4lymphocyte_percent']
sequential_features_t5=['CRT5lymphocyte_absolute_count_KuL','CRT5lymphocyte_percent']

sequential_features= sequential_features_t1+sequential_features_t2+\
                     sequential_features_t3+sequential_features_t4+\
                     sequential_features_t5
def string_float(data):
    df= copy.deepcopy(data)
    columns= data.shape[1]
    for col in range(columns):
        tmp=list(map(lambda x: 0 if x==' ' else float(x),data.iloc[:,col].tolist()))
        median= np.median(tmp)
        df.iloc[:,col]= list(map(lambda x: median if x==0 else x,tmp))
    return df

def create_train_data(df_train):
    y_train_class= df_train.pop('G4RIL').values
    y_train_reg= df_train.pop('CRT_ALCnadir').values
    x_train_dense = MinMaxScaler((0, MAX_VALUE)).fit_transform(string_float(df_train[dense_features]).values)
    x_train_sparse = MinMaxScaler((0, MAX_VALUE)).fit_transform(string_float(df_train[sparse_features]).values)
    x_train_init_seq = MinMaxScaler((0, MAX_VALUE)).fit_transform(string_float(df_train[sequential_features_t0]).values)
    y_train_sequential = MinMaxScaler((0, MAX_VALUE)).fit_transform(string_float(df_train[sequential_features]).values)
    return x_train_dense, x_train_sparse,x_train_init_seq,y_train_sequential

def create_test_data(df_test):
    y_test_class = df_test.pop('G4RIL').values
    y_test_reg = df_test.pop('CRT_ALCnadir').values
    x_test_dense = MinMaxScaler((0, MAX_VALUE)).fit_transform(string_float(df_test[dense_features]).values)
    x_test_sparse = MinMaxScaler((0, MAX_VALUE)).fit_transform(string_float(df_test[sparse_features]).values)
    x_test_init_seq = MinMaxScaler((0, MAX_VALUE)).fit_transform(string_float(df_test[sequential_features_t0]).values)
    y_test_sequential = MinMaxScaler((0, MAX_VALUE)).fit_transform(string_float(df_test[sequential_features]).values)
    return x_test_dense,x_test_sparse,x_test_init_seq,y_test_sequential













