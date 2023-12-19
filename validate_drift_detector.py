from datetime import timedelta
from pathlib import Path
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy.random import seed
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.utils import set_random_seed
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from mmd import mmd_linear, mmd_rbf, mmd_poly
from utils import path_to_features, crop_data, pad_data, create_base_model, evaluate_model, detect_cusum
from utils import get_cambridge_data as read_data

import pandas as pd
import tensorflow as tf
import numpy as np

kseed = 1
seed(kseed)
set_random_seed(kseed)
tf.config.experimental.enable_op_determinism()
tqdm.pandas()
model_path = 'Models' / Path('cambridge') / 'drift_detector'
if not model_path.exists():
    model_path.mkdir(parents=True)

data = read_data()
date70 = data.record_date.sort_values().iloc[int(len(data) * 0.7)]
data['features'] = data.progress_apply(path_to_features, axis=1)

development_data = data[data.record_date < date70]

date70 = development_data.record_date.sort_values().iloc[int(len(development_data) * 0.7)]

post_development_data = development_data[development_data.record_date >= date70]
development_data = development_data[development_data.record_date < date70]
post_development_data = post_development_data[~(post_development_data.uid.isin(development_data.uid))]

train = []
valid = []
test = []
for appearances in development_data.uid.value_counts(sort=True).unique():
    indices = development_data.uid.value_counts()[development_data.uid.value_counts() == appearances].index
    trn, rem = train_test_split(development_data[development_data.uid.isin(indices)].uid.unique(), test_size=0.4,
                                stratify=development_data[development_data.uid.isin(indices)].drop_duplicates('uid').COVID_STATUS, random_state=kseed)
    vld, tst = train_test_split(rem, test_size=0.5, stratify=development_data[development_data.uid.isin(rem)].drop_duplicates('uid').COVID_STATUS, random_state=kseed)
    train.extend(trn)
    valid.extend(vld)
    test.extend(tst)

train = development_data[development_data.uid.isin(train)]
valid = development_data[development_data.uid.isin(valid)]
test = development_data[development_data.uid.isin(test)]

length = int(train['features'].apply(lambda x: x.shape[0]).quantile(0.90))

X_train, Y_train = np.array(train['features'].progress_apply(lambda x: crop_data(pad_data(x, length))).to_list()), np.array(train['COVID_STATUS'].to_list())

X_test, Y_test = np.array(test['features'].progress_apply(lambda x: crop_data(pad_data(x, length))).to_list()), np.array(test['COVID_STATUS'].to_list())

X_valid, Y_valid = np.array(valid['features'].progress_apply(lambda x: crop_data(pad_data(x, length))).to_list()), np.array(valid['COVID_STATUS'].to_list())

X_post, Y_post = np.array(post_development_data['features'].progress_apply(lambda x: crop_data(pad_data(x, length))).to_list()), np.array(
    post_development_data['COVID_STATUS'].to_list())

clbck = [ModelCheckpoint(model_path / 'model{epoch:02d}.hdf5', save_weights_only=True, verbose=1),
         EarlyStopping(monitor="val_auc", patience=25, verbose=1, mode="max", restore_best_weights=True)]
model = create_base_model(X_train.shape[1:])
h = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), verbose=1, batch_size=32, epochs=100, callbacks=clbck)

thrs = evaluate_model(Y_valid, model.predict(X_valid, verbose=1))

_ = evaluate_model(Y_test, model.predict(X_test, verbose=1), thrs)

_ = evaluate_model(Y_post, model.predict(X_post, verbose=1), thrs)

m = Model(model.inputs[0], model.get_layer('tf.math.reduce_mean').output)

preds = model.predict(np.array(post_development_data.sort_values('record_date')['features'].apply(lambda x: crop_data(pad_data(x))).to_list()), verbose=1)
test_preds = (model.predict(X_test) > thrs) * 1
y_true = np.array(post_development_data.sort_values('record_date')['COVID_STATUS'].to_list())

post_dev_embeddings = m.predict(np.array(post_development_data.sort_values('record_date')['features'].apply(lambda x: crop_data(pad_data(x))).to_list()))
dev_embeddings = m.predict(np.array(train['features'].apply(lambda x: crop_data(pad_data(x))).to_list()), verbose=1)
dev_positive_embeddings = m.predict(np.array(train[train.COVID_STATUS == 1]['features'].apply(lambda x: crop_data(pad_data(x))).to_list()), verbose=1)
dev_negative_embeddings = m.predict(np.array(train[train.COVID_STATUS != 1]['features'].apply(lambda x: crop_data(pad_data(x))).to_list()), verbose=1)

l = []
for win_len in tqdm([7, 10, 14]):
    for interval in set([round(win_len * i) for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]):
        start_date = post_development_data.record_date.dt.normalize().sort_values().min()
        end_date = post_development_data.record_date.dt.normalize().sort_values().max() + timedelta(days=interval)
        ind = [post_development_data.sort_values('record_date').reset_index()[((start_date + timedelta(days=i * interval) <= post_development_data.sort_values('record_date').record_date) & (
                    post_development_data.sort_values('record_date').record_date <= (start_date + timedelta(days=i * interval) + timedelta(days=win_len)))).reset_index(drop=True)].index for i in
               range((end_date - start_date - timedelta(days=win_len)).days // interval + 1)]
        ind = [i for i in ind if len(i)]
        for min_batch_size in [min([len(i) for i in ind])] + list(range((1 + min([len(i) for i in ind]) // 10) * 10, max([len(i) for i in ind]), 10)):
            indices = [(i.append(pd.Index(list(range(i.min() - (min_batch_size - len(i)), i.min())))).sort_values()) if len(i) < min_batch_size else i for i in ind]
            indices = [(i - i.min() if (i.min() < 0) else i) for i in indices]
            if len(indices) > 5:
                performance_series = [balanced_accuracy_score(y_true[i], (preds[i] > thrs) * 1) for i in indices]
                ground_truth_concept_drift = (performance_series < balanced_accuracy_score(Y_test, test_preds)) * 1
                for ref in ['dev', 'positive', 'negative', 'dev-positive', 'dev-negative', 'positive-negative', 'dev-positive-negative']:
                    for kernel in ['linear', 'gaussian', 'polynomial']:
                        mmd = (mmd_linear if kernel == 'linear' else (mmd_rbf if kernel == 'gaussian' else (mmd_poly if kernel == 'polynomial' else None)))
                        distance = [((mmd(dev_embeddings, post_dev_embeddings[i]) if 'dev' in ref else 0) + (
                            mmd(dev_positive_embeddings, post_dev_embeddings[i]) if 'positive' in ref else 0) + (
                                         mmd(dev_negative_embeddings, post_dev_embeddings[i]) if 'negative' in ref else 0)) for i in indices]
                        for drift in [0.2, 0.3, 0.4, 0.5]:
                            for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                                alarm_end, alarm_start, _, _ = detect_cusum(distance, threshold, drift)
                                if alarm_end.size:
                                    perf_det, dist_det = np.array([[], []], dtype=float)
                                    pred_concept_drift = np.zeros(len(performance_series))
                                    samps = []
                                    for i, j in enumerate(alarm_start):
                                        perf_det = np.append(perf_det, performance_series[j:alarm_end[i] + 1])
                                        dist_det = np.append(dist_det, distance[j:alarm_end[i] + 1])
                                        pred_concept_drift[j:alarm_end[i] + 1] = 1

                                        union_index = pd.Index(data=[], dtype='Int64')
                                        for index in indices[j:alarm_end[i] + 1]:
                                            union_index = union_index.append(index)

                                        samps.append(union_index.unique())

                                    correlation, p_value = pearsonr(perf_det, dist_det) if len(perf_det) > 1 else (0, 1)
                                    perf_dif = balanced_accuracy_score(Y_test, test_preds) - balanced_accuracy_score(y_true[np.unique(np.concatenate(samps))],
                                                                                                                     (preds[np.unique(np.concatenate(samps))] > thrs) * 1)
                                    l.append([win_len, interval, min_batch_size, drift, threshold, ref, kernel, alarm_start, alarm_end, correlation, p_value, samps,
                                              sum(len(s) for s in samps), perf_dif, ground_truth_concept_drift, pred_concept_drift])

columns = ['win_len', 'interval', 'min_batch_size', 'drift', 'threshold', 'ref', 'kernel', 'alarm_start', 'alarm_end', 'correlation', 'p_value', 'samps', 'len_samps', 'perf_dif',
           'ground_truth_concept_drift', 'pred_concept_drift']
df = pd.DataFrame(l, columns=columns)

df.to_pickle(model_path / 'drift_detector_validation_results.pickle')
