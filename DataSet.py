import torch
import glob
import os
import pandas as pd
import numpy as np

class MapStyleDataset(torch.utils.data.Dataset):
    def __init__(self,data_path='./Data',file_name='cancer.csv',**kwargs):
        super(MapStyleDataset).__init__(**kwargs)
        full_path = os.path.join(data_path,file_name)


        dense_features1 = ['Age', 'BMI', 'CRT0ALC', 'Total_blood_volume_litres_Nadlerformula']
        dense_features2 = ['PTV', 'bodyV5_rel', 'bodyV10_rel', 'bodyV15_rel',
                           'bodyV20_rel', 'bodyV25_rel', 'bodyV30_rel', 'bodyV35_rel', 'bodyV40_rel',
                           'bodyV45_rel', 'bodyV50_rel', 'meanbodydose', 'bodyvolume', 'lungV5_rel',
                           'lungV10_rel', 'lungV15_rel', 'lungV20_rel', 'lungV25_rel', 'lungV30_rel',
                           'lungV35_rel', 'lungV40_rel', 'lungV45_rel', 'lungV50_rel', 'meanlungdose',
                           'lungvolume', 'heartV5_rel', 'heartV10_rel', 'heartV15_rel', 'heartV20_rel',
                           'heartV25_rel', 'heartV30_rel', 'heartV35_rel', 'heartV40_rel', 'heartV45_rel',
                           'heartV50_rel', 'meanheartdose', 'heartvolume', 'spleenV5_rel', 'spleenV10_rel',
                           'spleenV15_rel', 'spleenV20_rel', 'spleenV25_rel', 'spleenV30_rel', 'spleenV35_rel',
                           'spleenV40_rel', 'spleenV45_rel', 'spleenV50_rel', 'meanspleendose', 'spleenvolume'
                           ]
        self.dense_features = dense_features1 + dense_features2
        self.sparse_features = ['IMRT1Protons0', 'Sex', 'Race', 'Histology',
                           'Location_uppmid_vs_low', 'Location_upp_vs_mid_vs_low', 'Induction_chemo',
                           'CChemotherapy_type']
        self.sequential_features_t0 = [
            'CRT0neutrophil_percent', 'CRT0lymphocyte_percent', 'CRT0monocyte_percent'
        ]

        sequential_features_t1 = ['CRT1lymphocyte_absolute_count_KuL', 'CRT1lymphocyte_percent']
        sequential_features_t2 = ['CRT2lymphocyte_absolute_count_KuL', 'CRT2lymphocyte_percent']
        sequential_features_t3 = ['CRT3lymphocyte_absolute_count_KuL', 'CRT3lymphocyte_percent']
        sequential_features_t4 = ['CRT4lymphocyte_absolute_count_KuL', 'CRT4lymphocyte_percent']
        sequential_features_t5 = ['CRT5lymphocyte_absolute_count_KuL', 'CRT5lymphocyte_percent']

        self.sequential_features = sequential_features_t1 + sequential_features_t2 + \
                              sequential_features_t3 + sequential_features_t4 + \
                              sequential_features_t5

        self.data = pd.read_csv(full_path, header=0, usecols=self.dense_features+
                                                             self.sparse_features+
                                                             self.sequential_features_t0+
                                                             self.sequential_features)
        self.stats = self.data.describe().loc['50%', :].tolist()
    def __len__(self):
        return len(self.data)

    def transform(self,row):
        for i in range(len(row)):
            if row[i]== ' ':
                row[i]=0
            else:
                row[i]= float(row[i])
        return row.astype(float)


    def __getitem__(self, item):
        len_x_d= len(self.dense_features)
        len_x_s= len(self.sparse_features)
        len_x_ini= len(self.sequential_features_t0)
        len_y_targ= len(self.sequential_features)
        return self.transform(self.data.iloc[item,0:len_x_d].values),\
               self.transform(self.data.iloc[item,len_x_d:len_x_d+len_x_s].values),\
               self.transform(self.data.iloc[item,len_x_d+len_x_s:len_x_d+len_x_s+len_x_ini].values), \
               self.transform(self.data.iloc[item, len_x_d+len_x_s+len_x_ini:].values)



class IterStyleDataSet(torch.utils.data.IterableDataset):
    def __init__(self,data_path='./Data',file_name='cancer.csv',**kwargs):
        super(IterStyleDataSet,self).__init__(**kwargs)
        full_path = os.path.join(data_path, file_name)
        dense_features1 = ['Age', 'BMI', 'CRT0ALC', 'Total_blood_volume_litres_Nadlerformula']
        dense_features2 = ['PTV', 'bodyV5_rel', 'bodyV10_rel', 'bodyV15_rel',
                           'bodyV20_rel', 'bodyV25_rel', 'bodyV30_rel', 'bodyV35_rel', 'bodyV40_rel',
                           'bodyV45_rel', 'bodyV50_rel', 'meanbodydose', 'bodyvolume', 'lungV5_rel',
                           'lungV10_rel', 'lungV15_rel', 'lungV20_rel', 'lungV25_rel', 'lungV30_rel',
                           'lungV35_rel', 'lungV40_rel', 'lungV45_rel', 'lungV50_rel', 'meanlungdose',
                           'lungvolume', 'heartV5_rel', 'heartV10_rel', 'heartV15_rel', 'heartV20_rel',
                           'heartV25_rel', 'heartV30_rel', 'heartV35_rel', 'heartV40_rel', 'heartV45_rel',
                           'heartV50_rel', 'meanheartdose', 'heartvolume', 'spleenV5_rel', 'spleenV10_rel',
                           'spleenV15_rel', 'spleenV20_rel', 'spleenV25_rel', 'spleenV30_rel', 'spleenV35_rel',
                           'spleenV40_rel', 'spleenV45_rel', 'spleenV50_rel', 'meanspleendose', 'spleenvolume'
                           ]
        self.dense_features = dense_features1 + dense_features2
        self.sparse_features = ['IMRT1Protons0', 'Sex', 'Race', 'Histology',
                                'Location_uppmid_vs_low', 'Location_upp_vs_mid_vs_low', 'Induction_chemo',
                                'CChemotherapy_type']
        self.sequential_features_t0 = [
            'CRT0neutrophil_percent', 'CRT0lymphocyte_percent', 'CRT0monocyte_percent'
        ]

        sequential_features_t1 = ['CRT1lymphocyte_absolute_count_KuL', 'CRT1lymphocyte_percent']
        sequential_features_t2 = ['CRT2lymphocyte_absolute_count_KuL', 'CRT2lymphocyte_percent']
        sequential_features_t3 = ['CRT3lymphocyte_absolute_count_KuL', 'CRT3lymphocyte_percent']
        sequential_features_t4 = ['CRT4lymphocyte_absolute_count_KuL', 'CRT4lymphocyte_percent']
        sequential_features_t5 = ['CRT5lymphocyte_absolute_count_KuL', 'CRT5lymphocyte_percent']

        self.sequential_features = sequential_features_t1 + sequential_features_t2 + \
                                   sequential_features_t3 + sequential_features_t4 + \
                                   sequential_features_t5

        self.data = pd.read_csv(full_path, header=0, usecols=self.dense_features +
                                                             self.sparse_features +
                                                             self.sequential_features_t0 +
                                                             self.sequential_features)

    def transform(self,row):
        for i in range(len(row)):
            if row[i]== ' ':
                row[i]=0
            else:
                row[i]= float(row[i])
        return row.astype(float)

    def __iter__(self):
        len_x_d = len(self.dense_features)
        len_x_s = len(self.sparse_features)
        len_x_ini = len(self.sequential_features_t0)
        # len_y_targ = len(self.sequential_features)
        for row in self.data.values:
            yield self.transform(row[0:len_x_d]), \
                  self.transform(row[len_x_d:len_x_d+len_x_s]),\
                  self.transform(row[len_x_d+len_x_s:len_x_d+len_x_s+len_x_ini]),\
                  self.transform(row[len_x_d+len_x_s+len_x_ini:])

    # def __iter__(self):
    #     len_x_d = len(self.dense_features)
    #     len_x_s = len(self.sparse_features)
    #     len_x_ini = len(self.sequential_features_t0)
    #     return iter(self.transform(self.data[0:len_x_d]))



