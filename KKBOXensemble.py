from itertools import combinations
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor, radians, sin, cos
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD, NMF, KernelPCA
import lightgbm as lgb

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical


def to_time(df, f_time='time'):
    
    df[f_time] = pd.to_datetime(df[f_time], unit='s')
    
    #numeric
    #f_mday = 'inf_scl_{}_day'.format(f_time)
    f_hour = 'inf_hour'
    f_wday = 'inf_wday'
    f_week = 'inf_week'
    f_wdhr = 'inf_wdhr'
    #f_week = 'inf_{}_week'.format(f_time)
        
    #d, h, m, w = 31, 24, 60, 7
    #df[f_mday] = df[f_time].dt.day# /d
    df[f_hour] = df[f_time].dt.hour# /h
    df[f_wday] = df[f_time].dt.dayofweek# /w
    df[f_week] = df[f_time].dt.week
    df[f_wdhr] = df[f_wday] * 24 + df[f_hour]
    df[f_wdhr] = df[f_wdhr].apply(str)
    
    #print(df.describe())

#string
def titles_agg(train_data, test_data, hist, stem='tmp', last_only=False):
    
    print('{}:\t{} records'.format(stem, hist.shape[0]), flush=True)
    col = 'list_ttl_{}'.format(stem)
    #list and count
    if last_only:
        col = 'list_ttl_{}_last_only'.format(stem)
        tmp = hist.groupby('user_id')['title_id'].agg(' '.join).apply(lambda x: x.split()[-1])
        
    else:    
        col = 'list_ttl_{}'.format(stem)
        tmp = hist.groupby('user_id')['title_id'].agg(' '.join)#.apply(lambda x: x.split())

    tmp = tmp.rename(col).to_frame()
    tmp['user_id'] = tmp.index
    tmp = tmp.reset_index(drop=True)
    
    train_data = train_data.merge(tmp, how='left', on='user_id')
    test_data = test_data.merge(tmp, how='left', on='user_id')

    train_data = train_data.fillna('')
    test_data = test_data.fillna('')

    if last_only:
        del tmp
        col = 'f_time_lastest_{}_last_only'.format(stem)
        tmp = hist.groupby('user_id')['watch_time'].agg(lambda x: ' '.join(str(x))).apply(lambda x: x.split()[-1])
    
        tmp = tmp.rename(col).to_frame()
        tmp['user_id'] = tmp.index
        tmp = tmp.reset_index(drop=True)

        train_data = train_data.merge(tmp, how='left', on='user_id')
        test_data = test_data.merge(tmp, how='left', on='user_id')

    else:
        train_data['f_cnt_{}'.format(stem)] = train_data[col].apply(lambda x: len(x.split()))
        test_data['f_cnt_{}'.format(stem)] = test_data[col].apply(lambda x: len(x.split()))
    
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
 
    #median time
    tmp = hist.groupby('user_id')['watch_time'].median()
    tmp = tmp.rename('f_time_median_{}'.format(stem)).to_frame()
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


#evaluation
def display_val_score(y, p, r):
    v = np.argmax(p, axis=1)
    jcc = jaccard_similarity_score(y, v)
    acc = accuracy_score(y, v)
    print('\nVal: jcc={:.6f}, acc={:.6f}'.format(jcc, acc), flush=True)
    print('Adjusted Val: jcc={:.6f}, acc={:.6f}'.format(jcc * ratio, acc * ratio), flush=True) 
    return jcc

#
def write_csv(test_id, labels, t='t', stem='', score=0):
    print("\nWriting output...\n")
    sub = pd.DataFrame()
    sub['user_id'] = test_id
    sub['title_id'] = labels
    print(sub['title_id'].value_counts())
    sub.to_csv("preds_{}_{}_s{:.6f}.csv".format(stem, t, jcc * ratio), index=False)


#read
input_folder = '../input/'
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
sel = train_users['title_id'].value_counts()
#print(sel)
#for i in range(100):
#    tmp = sel.loc[sel >= i].index.tolist()
#    users = train_users.loc[(train_users['title_id'].isin(tmp))]
#    print('{}: {}, {} ({:.6f}, {:.6f})'.format(i, len(tmp), len(users), len(users)/total, i/total), flush=True)    
    
min_hits = 7 #min1
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

s = 'lastest'
train_users, test_users = titles_agg(train_users, test_users, all_events, stem=s, last_only=True)
postfix_stem = 'list_ttl_{}_last_only'.format(s)

#print(train_users)

#short=>dislike
t = 60 * 5 #watch_time
s = 'in{:04d}s'.format(t)
sel_events = all_events.loc[(all_events['watch_time'] <= t)]
train_users, test_users = titles_agg(train_users, test_users, sel_events, stem=s)

#medium
t = 60 * 3 #watch_time
m = 60 * 50
s = 'out{:04d}s{:04d}s'.format(t, m)
sel_events = all_events.loc[(all_events['watch_time'] >= t) & (all_events['watch_time'] <= m)]
train_users, test_users = titles_agg(train_users, test_users, sel_events, stem=s)

#long
t = 60 * 40 #watch_time
s = 'out{:04d}s'.format(t)
sel_events = all_events.loc[(all_events['watch_time'] >= t)]
train_users, test_users = titles_agg(train_users, test_users, sel_events, stem=s)

#lastest-1
#recent intested in
w = 39 - 1 #w-th week
t = 60 * 3 #watch_time
s = 'out{:04d}s{}w'.format(t, w)
sel_events = all_events.loc[(all_events['watch_time'] >= t) & (all_events['inf_week'] >= w)]
train_users, test_users = titles_agg(train_users, test_users, sel_events, stem=s)
train_users, test_users = sum_watch_time(train_users, test_users, sel_events, stem=s)
train_users, test_users = trigger_time(train_users, test_users, all_events, stem=s)

print(train_users.shape)


#features list
f_ttl = [s for s in train_users.columns.tolist() if s.startswith('list_ttl')]
print('{}: {}'.format(len(f_ttl), f_ttl))
f_trg = [s for s in train_users.columns.tolist() if s.startswith('list_trg')]
print('{}: {}'.format(len(f_trg), f_trg))
f_num = [s for s in train_users.columns.tolist() if s.startswith('f_')]
print('{}: {}'.format(len(f_num), f_num))

#dataset
target_lbl = LabelEncoder()
candidates = train_users['title_id'].tolist() + train_users[postfix_stem].tolist() + test_users[postfix_stem].tolist()
candidates = target_lbl.fit_transform(candidates)
train_y = target_lbl.transform(train_users['title_id'].tolist())
#y_max = max(train_y) + 1
y_max = max(candidates) + 1
print(train_y.shape)
#positx
train_postfix = target_lbl.transform(train_users[postfix_stem].tolist())
test_postfix = target_lbl.transform(test_users[postfix_stem].tolist())

#numerics
for f in f_num:
    train_users[f] = train_users[f].apply(np.nan_to_num)
    test_users[f] = test_users[f].apply(np.nan_to_num)
    #print(train_users[f])
    
scalar = MinMaxScaler(feature_range=(0, 1), copy=True)
train_users[f_num] = scalar.fit_transform(train_users[f_num])
test_users[f_num] = scalar.transform(test_users[f_num])

train_X_num = train_users[f_num].as_matrix()
test_X_num = test_users[f_num].as_matrix()


train_X = [train_X_num]
test_X = [test_X_num]

#CountVec Merged
ttl_cnt = len(list(all_events['title_id'].unique()))
cntVec = CountVectorizer(ngram_range=(1, 1), analyzer='word')
cntVec.fit(all_events['title_id'])
#cntVec.fit(candidates)
for f in f_ttl:
    add = cntVec.transform(train_users[f])
    add = np.log1p(add)
    #train_X = sparse.hstack((train_X, add)).todense()
    train_X.append(add.todense())
    print('{} +{}'.format(f, add.shape[1]), flush=True)
    #del add
    #ttl_cnt = add.todense().shape[1]
    
    add = cntVec.transform(test_users[f])
    add = np.log1p(add)
    #test_X = sparse.hstack((test_X, add)).todense()
    test_X.append(add.todense())
    #del add

#CountVec Merged
#wdhr = len(list(all_events['inf_wdhr'].unique()))
cntVec = CountVectorizer(ngram_range=(1, 1), analyzer='word')
cntVec.fit(all_events['inf_wdhr'])
for f in f_trg:
    add = cntVec.transform(train_users[f])
    add = np.log1p(add)
    #train_X = sparse.hstack((train_X, add)).todense()
    train_X.append(add.todense())
    print('{} +{}'.format(f, add.shape[1]), flush=True)
    #del add
    
    add = cntVec.transform(test_users[f])
    add = np.log1p(add)
    #test_X = sparse.hstack((test_X, add)).todense()
    test_X.append(add.todense())
    #del add
    
    wdhr = add.todense().shape[1]

print('\ndims for each feature', flush=True)
inputs_ndim = []
for x in train_X:
    print(x.shape, flush=True)
    inputs_ndim.append(x.shape[1])

#fold for CV
print('Assigning CV', flush=True)
nr_splits = 7
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
#postfix
postfix_valid = train_postfix[valid_sets[0]]

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
num_dence = Dense(32, activation='relu')(num_dence_input)

inputs_collected.append(num_dence_input)
dense_collected.append(num_dence)

#shared dense
dense_ttl = Dense(16, activation='relu')#16 * 6
dense_wdhr = Dense(8, activation='relu')

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
dense_ds1 = Dense(256, activation='relu')(dense_dp1)
dense_dp2 = Dropout(0.5)(dense_ds1)
output = Dense(y_max, activation='softmax')(dense_dp2)

model = Model(inputs=inputs_collected, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


train_keras = True
#train_keras = False

if train_keras:
    print(model.summary(), flush=True)
    print('Training keras', flush=True)

    #callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    bst_model_path = tmstmp + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    #fit
    hist = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=128, shuffle=True, callbacks=[early_stopping, model_checkpoint])
    #        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
    
    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
    
    val_probs = model.predict(X_valid, batch_size=4096, verbose=1)
    jcc = display_val_score(y=y_valid, p=val_probs, r=ratio)    
    
    # make the submission
    print('\n\nPrediction', flush=True)
    probs = model.predict(test_X, batch_size=4096, verbose=1)
    #
    preds = target_lbl.inverse_transform(np.argmax(probs, axis=1))
    write_csv(test_id=test_users['user_id'], labels=preds, t=tmstmp, stem='keras', score=jcc * ratio)
else:
    val_probs = np.zeros((X_valid[0].shape[0], y_max))
    probs = np.zeros((test_users.shape[0], y_max))


#sklearn
X_train = np.nan_to_num(np.array(np.concatenate(X_train, axis=1)))
X_valid = np.nan_to_num(np.array(np.concatenate(X_valid, axis=1)))
print(X_train.shape, X_valid.shape)
test_X = np.nan_to_num(np.array(np.concatenate(test_X, axis=1)))

#rescale
scalar = MinMaxScaler(feature_range=(0, 1), copy=True)
X_train = scalar.fit_transform(X_train)
X_valid = scalar.transform(X_valid)
test_X = scalar.transform(test_X)


train_sklearn = True
#train_sklearn = False

if train_sklearn:
    print('\nGBM', flush=True)
    params = {}
    params['num_threads'] = 4
    #params['boost'] = 'gbdt'
    params['boost'] = 'dart'
    #params['num_class'] = 1
    #params['metric'] = 'multi_logloss'        
    #params['objective'] = 'multiclass'
    params['is_unbalance'] = True
    params['metric'] = 'binary_logloss'
    params['objective'] = 'binary'
    params['min_data_in_leaf'] = 2 ** 1 #default 100
    
    #learning
    params['learning_rate'] = 0.11
    params['num_leaves'] = 2 ** 5
    
    if params.get('boost') == 'dart':
        params['drop_rate'] = 0.25 #dart, deafault 0.1
        params['skip_drop'] = 0.75 #dart, deafault 0.5
        params['max_drop'] = 50 #dart, deafault 50
        params['uniform_drop'] = False #dart, deafault False
        params['xgboost_dart_mode'] = False #dart, deafault False
        #params['xgboost_dart_mode'] = True #dart, deafault False
        
    #params['min_hessian'] = 10.0 #default 10.0
    params['feature_fraction'] = 0.5 #default=1.0
    params['bagging_fraction'] = 0.7 #default=1.0
    params['bagging_freq'] = 3
    params['lambda_l1'] = 0.007 #default 0
    params['lambda_l2'] = 0.019 #default 0
    params['data_random_seed'] = 62017
    params['verbose'] = 0 #<0 = Fatel, =0 = Error(Warn), >0 = Info
    
    #metric
    params['metric_freq'] = 5 #deafult 1

    max_bin = 2 ** 13
    
    num_rounds, min_rounds = 250, 10
    #    
    sk_probs = np.zeros((X_valid.shape[0], y_max))
    test_probs = np.zeros((test_users.shape[0], y_max))    

    y_train_sparse = np.zeros((X_train.shape[0], y_max))
    for i, j in enumerate(y_train):
            y_train_sparse[i, j] = 1

    y_valid_sparse = np.zeros((X_valid.shape[0], y_max))
    for i, j in enumerate(y_valid):
            y_valid_sparse[i, j] = 1

    i = 0
    for c in range(y_max):
        if np.sum(y_train_sparse[:, c]) > 0:
            print('lightGBM w/ eta={} leaves={}'.format(params['learning_rate'], params['num_leaves']))
            dtrain = lgb.Dataset(X_train, label=y_train_sparse[:, c], weight=None, max_bin=max_bin, reference=None, free_raw_data=False)
            dvalid = lgb.Dataset(X_valid, label=y_valid_sparse[:, c], reference=X_train, free_raw_data=False)
            gbm = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid], valid_names=['tr', 'va'], 
                            num_boost_round=num_rounds, early_stopping_rounds=min_rounds)
    
            sk_probs[:, c] = gbm.predict(X_valid, num_iteration=gbm.best_iteration)[:]#[:, 1]
            test_probs[:, c] = gbm.predict(test_X, num_iteration=gbm.best_iteration)[:]#[:, 1]
            i += 1
            print('no{:04d}: {:04d}'.format(i, c), flush=True)
    
    jcc = display_val_score(y=y_valid, p=sk_probs, r=ratio)
     
    #
    preds = target_lbl.inverse_transform(np.argmax(test_probs, axis=1))
    write_csv(test_id=test_users['user_id'], labels=preds, t=tmstmp, stem='gbm', score=jcc* ratio)

    w = 0.8
    val_probs += sk_probs * w
    probs += test_probs * w

opt_postfix = True
#opt_postfix = False
if opt_postfix:
    print('\nPostFix Labels')
    max_iter = 1000
    fix, best_fix, best_jcc = 0.001, 0, 0
    for k in range(max_iter+1):
        #fixing
        eval_probs = val_probs.copy()
        for i, j in enumerate(postfix_valid):
            eval_probs[i, j] += fix * k
        
        #eval
        jcc = jaccard_similarity_score(y_valid, np.argmax(eval_probs, axis=1))
        if jcc > best_jcc:
            best_jcc = jcc
            best_fix = fix * k
            print('*current best jcc={:.6f} w/ fix={:.3f}'.format(best_jcc, best_fix), flush=True)
    
    print('Best jcc={:.6f} w/ fix={:.3f}'.format(best_jcc, best_fix), flush=True)
    print('Adjusted best jcc={:.6f} w/ fix={:.3f}'.format(best_jcc * ratio, best_fix), flush=True)
    jcc = best_jcc * ratio
    for i, j in enumerate(test_postfix):
        probs[i, j] += best_fix
        
    #make the submission
    print('\n\nPrediction', flush=True)
    preds = target_lbl.inverse_transform(np.argmax(probs, axis=1))
    write_csv(test_id=test_users['user_id'], labels=preds, t=tmstmp, stem='keras_fix', score=jcc * ratio)

    
    
    
