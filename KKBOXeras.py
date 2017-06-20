# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:24:59 2017

@author: Tsai, Chia-Ta
"""
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor
from random import choice, sample, uniform
import time
#pyData stack
import numpy as np
import pandas as pd
from scipy import sparse
#sklearn preprocessing, model selection
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
#sklearn classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import jaccard_similarity_score, accuracy_score

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint



def to_time(df, f_time='time'):
    
    df[f_time] = pd.to_datetime(df[f_time], unit='s')
    
    #numeric
    f_hour = 'inf_hour'
    f_wday = 'inf_wday'
    f_week = 'inf_week'
    f_wdhr = 'inf_wdhr'
        
    #d, h, m, w = 31, 24, 60, 7
    df[f_hour] = df[f_time].dt.hour
    df[f_wday] = df[f_time].dt.dayofweek
    df[f_week] = df[f_time].dt.week
    df[f_wdhr] = df[f_wday] * 24 + df[f_hour]
    df[f_wdhr] = df[f_wdhr].apply(str)
    
    #print(df.describe())

#string
def titles_agg(train_data, test_data, hist, stem='tmp'):
    
    print('{}:\t{} records'.format(stem, hist.shape[0]), flush=True)
    #list and count
    tmp = hist.groupby('user_id')['title_id'].agg(' '.join)#.apply(lambda x: x.split())
    tmp = tmp.rename('list_ttl_{}'.format(stem)).to_frame()
    tmp['user_id'] = tmp.index
    tmp = tmp.reset_index(drop=True)
    
    train_data = train_data.merge(tmp, how='left', on='user_id')
    train_data = train_data.fillna('')
    train_data['f_cnt_{}'.format(stem)] = train_data['list_ttl_{}'.format(stem)].apply(lambda x: len(x.split()))
        
    test_data = test_data.merge(tmp, how='left', on='user_id')
    test_data = test_data.fillna('')
    test_data['f_cnt_{}'.format(stem)] = test_data['list_ttl_{}'.format(stem)].apply(lambda x: len(x.split()))
    
    del tmp
    return train_data, test_data

#int
def sum_watch_time(train_data, test_data, hist, stem='tmp'):
    
    #sum time
    tmp = hist.groupby('user_id')['watch_time'].sum()
    tmp = tmp.rename('f_time_sum_{}'.format(stem)).to_frame()
    tmp['user_id'] = tmp.index
    tmp = tmp.reset_index(drop=True)
        
    #merge
    train_data = train_data.merge(tmp, how='left', on='user_id')   
    test_data = test_data.merge(tmp, how='left', on='user_id')
    del tmp

    #var time
    tmp = hist.groupby('user_id')['watch_time'].var()
    tmp = tmp.rename('f_time_var_{}'.format(stem)).to_frame()
    tmp['user_id'] = tmp.index
    tmp = tmp.reset_index(drop=True)
        
    #merge
    train_data = train_data.merge(tmp, how='left', on='user_id')   
    test_data = test_data.merge(tmp, how='left', on='user_id')
    del tmp
    
    train_data = train_data.fillna(0) 
    test_data = test_data.fillna(0)

    #print(train_data)
    return train_data, test_data

#string
def trigger_time(train_data, test_data, hist, stem='tmp'):

    tmp = hist.groupby('user_id')['inf_wdhr'].agg(' '.join)#.apply(lambda x: x.split())
    tmp = tmp.rename('list_trg_{}'.format(stem)).to_frame()
    tmp['user_id'] = tmp.index
    tmp = tmp.reset_index(drop=True)

    #merge
    train_data = train_data.merge(tmp, how='left', on='user_id')
    train_data = train_data.fillna('')
    train_data['f_cnt_{}'.format(stem)] = train_data['list_trg_{}'.format(stem)].apply(lambda x: len(x.split()))
        
    test_data = test_data.merge(tmp, how='left', on='user_id')
    test_data = test_data.fillna('')
    test_data['f_cnt_{}'.format(stem)] = test_data['list_trg_{}'.format(stem)].apply(lambda x: len(x.split()))
  

    del tmp
    return train_data, test_data


#read
input_folder = './'
####train
train_events = pd.read_csv(input_folder + 'events_train.csv', dtype={'user_id': np.str, 'title_id': np.str})
train_users = pd.read_csv(input_folder + 'labels_train.csv', dtype={'user_id': np.str, 'title_id': np.str})
####test
test_events = pd.read_csv(input_folder + 'events_test.csv', dtype={'user_id': np.str, 'title_id': np.str})
test_users = pd.DataFrame()
test_users['user_id'] = test_events['user_id'].unique()

#use top titles from both train and test;   
all_events = pd.concat([train_events, test_events]).reset_index(drop=True)
to_time(all_events)

#clearing labels
total = len(train_users)
  
min_hits = 5
sel = train_users['title_id'].value_counts()
print('Existing {} Labels'.format(len(sel)))
sel = sel.loc[sel >= min_hits].index.tolist()
print('Reduced to {} Labels, removing minors less freq <= {}'.format(len(sel), min_hits), flush=True)
train_users = train_users.loc[(train_users['title_id'].isin(sel))]
ratio = len(train_users) / total
print('Ratio = {:.6f}\n'.format(ratio), flush=True)


#all
s = 'overall'
train_users, test_users = titles_agg(train_users, test_users, all_events, stem=s)
train_users, test_users = sum_watch_time(train_users, test_users, all_events, stem=s)
train_users, test_users = trigger_time(train_users, test_users, all_events, stem=s)

#rough
#short=>dislike
t = 60 * 5 #watch_time
s = 'in{:04d}s'.format(t)
sel_events = all_events.loc[all_events['watch_time'] <= t]
train_users, test_users = titles_agg(train_users, test_users, sel_events, stem=s)

###########
#lastest-1
#recent intested in
w = 39 #w-th week
t = 60 * 5 #watch_time
s = 'out{:04d}s{}w'.format(t, w)
sel_events = all_events.loc[(all_events['watch_time'] >= t) & (all_events['inf_week'] >= w)]
train_users, test_users = titles_agg(train_users, test_users, sel_events, stem=s)
train_users, test_users = sum_watch_time(train_users, test_users, sel_events, stem=s)

print(train_users.shape)

#features list
print('Extracted features:')
f_ttl = [s for s in train_users.columns.tolist() if s.startswith('list_ttl')]
print('{}: {}'.format(len(f_ttl), f_ttl))
f_trg = [s for s in train_users.columns.tolist() if s.startswith('list_trg')]
print('{}: {}'.format(len(f_trg), f_trg))
f_num = [s for s in train_users.columns.tolist() if s.startswith('f_')]
print('{}: {}'.format(len(f_num), f_num))

#dataset
target_lbl = LabelEncoder()
train_y = target_lbl.fit_transform(train_users['title_id'].tolist())
y_max = max(train_y) + 1
print(train_y.shape)

#numerics
for f in f_num:
    train_users[f] = train_users[f].apply(np.nan_to_num)
    test_users[f] = test_users[f].apply(np.nan_to_num)
    
scalar = MinMaxScaler(feature_range=(0, 1), copy=True)
train_users[f_num] = scalar.fit_transform(train_users[f_num])
test_users[f_num] = scalar.transform(test_users[f_num])

train_X_num = train_users[f_num].as_matrix()
test_X_num = test_users[f_num].as_matrix()


train_X = [train_X_num]
test_X = [test_X_num]

ttl_cnt = len(list(all_events['title_id'].unique()))
#CountVec Merged
cntVec = CountVectorizer(ngram_range=(1, 1), analyzer='word')
cntVec.fit(all_events['title_id'])
for f in f_ttl:
    add = cntVec.transform(train_users[f])
    add = np.log1p(add)
    train_X.append(add.todense())
    print('{} +{}'.format(f, add.shape[1]), flush=True)
    
    add = cntVec.transform(test_users[f])
    add = np.log1p(add)
    test_X.append(add.todense())

#CountVec Merged
cntVec = CountVectorizer(ngram_range=(1, 1), analyzer='word')
cntVec.fit(all_events['inf_wdhr'])
for f in f_trg:
    add = cntVec.transform(train_users[f])
    add = np.log1p(add)
    train_X.append(add.todense())
    print('{} +{}'.format(f, add.shape[1]), flush=True)
    
    add = cntVec.transform(test_users[f])
    add = np.log1p(add)
    test_X.append(add.todense())
    
    wdhr = add.todense().shape[1]

print('\ndims for each feature', flush=True)
inputs_ndim = []
for x in train_X:
    print(x.shape, flush=True)
    inputs_ndim.append(x.shape[1])

#fold for CV
print('Assigning CV', flush=True)
nr_splits = 5
fold_gen_seed = 62017
train_sets, valid_sets = list(), list()
fold_gen = StratifiedKFold(n_splits=nr_splits, shuffle=True, random_state=fold_gen_seed)
for train_indices, valid_indices in fold_gen.split(train_y, train_y):
    train_sets.append(train_indices)
    valid_sets.append(valid_indices)

X_train = []
X_valid = []
y_train = train_y[train_sets[0]]
y_valid = train_y[valid_sets[0]]
for x in train_X:
    X_train.append(x[train_sets[0]])
    X_valid.append(x[valid_sets[0]])


tmstmp = '{}'.format(time.strftime("%Y-%m-%d-%H-%M"))

# define the model structure
########################################
inputs_collected = []
dense_collected = []

num_dence_input = Input(shape=(inputs_ndim[0],))#, dtype='int32')

#ordinary dense
num_dence = Dense(16, activation='relu')(num_dence_input)

inputs_collected.append(num_dence_input)
dense_collected.append(num_dence)

#shared dense
dense_ttl = Dense(16, activation='relu')
dense_wdhr = Dense(4, activation='relu')

for x in inputs_ndim:
    #for titles
    if x == ttl_cnt:
        ttl_dence_input = Input(shape=(ttl_cnt,))#, dtype='int32')
        ttl_dence1 = dense_ttl(ttl_dence_input)
        
        inputs_collected.append(ttl_dence_input)
        dense_collected.append(ttl_dence1)
        
    #for wdhr
    if x == wdhr:
        wdhr_dence_input = Input(shape=(wdhr,))#, dtype='int32')
        wdhr_dence1 = dense_wdhr(wdhr_dence_input)
        
        inputs_collected.append(wdhr_dence_input)
        dense_collected.append(wdhr_dence1)        
        
concat = concatenate(dense_collected, axis=-1)

#final
dense_bn = BatchNormalization()(concat)
dense_dp1 = Dropout(0.25)(dense_bn)
dense_ds1 = Dense(512, activation='relu')(dense_dp1)
dense_dp2 = Dropout(0.5)(dense_ds1)
output = Dense(y_max, activation='softmax')(dense_dp2)

model = Model(inputs=inputs_collected, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

#print(model.summary(), flush=True)
print('Training keras', flush=True)

#callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = tmstmp + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

#fit
hist = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=128, shuffle=True, callbacks=[early_stopping, model_checkpoint])
    #        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
    
model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])
    
val_probs = model.predict(X_valid, batch_size=4096, verbose=1)
val_preds = np.argmax(val_probs, axis=1)
jcc = jaccard_similarity_score(y_valid, val_preds)
acc = accuracy_score(y_valid, val_preds)
print('\nVal: loss={:.6f}, jcc={:.6f}, acc={:.6f}'.format(bst_val_score, jcc, acc), flush=True)
    
# make the submission
print('\nPrediction', flush=True)
probs = model.predict(test_X, batch_size=4096, verbose=1)
    
#
print("\nWriting output...\n\n")
sub = pd.DataFrame()
sub['user_id'] = test_users['user_id']
sub['title_id'] = target_lbl.inverse_transform(np.argmax(probs, axis=1))
#sub['title_id'] = sub['title_id'].apply(lambda x: '{:08d}'.format(x))
print(sub['title_id'].value_counts())
sub.to_csv("preds_keras_{}_s{:.6f}.csv".format(tmstmp, jcc* ratio), index=False)

