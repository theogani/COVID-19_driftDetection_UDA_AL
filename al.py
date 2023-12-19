from datetime import timedelta
from pathlib import Path
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import set_random_seed
from numpy.random import seed
from tqdm import tqdm
from keras import Model
from keras.models import load_model
from mmd import mmd_linear, mmd_rbf, mmd_poly
from utils import crop_data, pad_data, detect_cusum, create_al_model

import numpy as np
import pandas as pd
import tensorflow as tf

kseed = 1
seed(kseed)
set_random_seed(kseed)
tf.config.experimental.enable_op_determinism()
tqdm.pandas()
model_path = Path('Models') / 'cambridge' / 'al_update_1'
if not model_path.exists():
    model_path.mkdir(parents=True)

post_development_data = pd.read_pickle(model_path.parent / 'post_development_data.pickle')
train = pd.read_pickle(model_path.parent / 'train.pickle')
test = pd.read_pickle(model_path.parent / 'test.pickle')
valid = pd.read_pickle(model_path.parent / 'valid.pickle')

length = int(train['features'].apply(lambda x: x.shape[0]).quantile(0.90))

model = load_model(model_path.parent / 'best_model.hdf5')
m = Model(model.inputs[0], model.get_layer('tf.math.reduce_mean').output)

post_dev_embeddings = m.predict(np.array(post_development_data.sort_values('record_date')['features'].apply(lambda x: crop_data(pad_data(x, length))).to_list()))
dev_embeddings = m.predict(np.array(train['features'].apply(lambda x: crop_data(pad_data(x, length))).to_list()), verbose=1)
dev_positive_embeddings = m.predict(np.array(train[train.COVID_STATUS == 1]['features'].apply(lambda x: crop_data(pad_data(x, length))).to_list()), verbose=1)
dev_negative_embeddings = m.predict(np.array(train[train.COVID_STATUS != 1]['features'].apply(lambda x: crop_data(pad_data(x, length))).to_list()), verbose=1)

df = pd.read_pickle(model_path.parent / 'drift_detector' / 'drift_detector_validation_results.pickle')

win_len, interval, min_batch_size, drift, threshold, ref, kernel = df.sort_values('concept_drift_balanced_accuracy', ascending=False).loc[
    df.concept_drift_balanced_accuracy == df.concept_drift_balanced_accuracy.max(), ['win_len', 'interval', 'min_batch_size', 'drift', 'threshold', 'ref', 'kernel']].values[0]
win_len = int(win_len)
interval = int(interval)

print(win_len, interval, min_batch_size, drift, threshold, ref, kernel)

start_date = post_development_data.record_date.dt.normalize().sort_values().min()
end_date = post_development_data.record_date.dt.normalize().sort_values().max() + timedelta(days=interval)

ind = [post_development_data.sort_values('record_date').reset_index()[(
        (start_date + timedelta(days=i * interval) <= post_development_data.sort_values('record_date').record_date) & (
        post_development_data.sort_values('record_date').record_date <= (start_date + timedelta(days=i * interval) + timedelta(days=win_len)))).reset_index(
    drop=True)].index for i in range((end_date - start_date - timedelta(days=win_len)).days // interval + 1)]

indices = [(i.append(pd.Index(list(range(i.min() - (min_batch_size - len(i)), i.min())))).sort_values()) if len(i) < min_batch_size else i for i in ind if len(i)]
indices = [(i - i.min() if (i.min() < 0) else i) for i in indices]
mmd = (mmd_linear if kernel == 'linear' else (mmd_rbf if kernel == 'gaussian' else (mmd_poly if kernel == 'polynomial' else None)))
distance = [((mmd(dev_embeddings, post_dev_embeddings[i]) if 'dev' in ref else 0) +
             (mmd(dev_positive_embeddings, post_dev_embeddings[i]) if 'positive' in ref else 0) +
             (mmd(dev_negative_embeddings, post_dev_embeddings[i]) if 'negative' in ref else 0)) for i in indices]
alarm_end, alarm_start, _, gp = detect_cusum(distance, threshold, drift)
ind = pd.Index(data=[], dtype='Int64')
for i in indices[alarm_start[0]:alarm_end[0] + 1]:
    ind = ind.append(i)

ind = ind.unique()

print(alarm_start, alarm_end, ind)

detected = post_development_data.sort_values('record_date').iloc[ind]
preds = model.predict(np.array(detected['features'].apply(lambda x: crop_data(pad_data(x))).to_list()))
detected = detected[np.logical_and(preds > (np.mean(preds) - np.std(preds)), preds < (np.mean(preds) + np.std(preds)))]

detected_train, detected_valid = train_test_split(detected, test_size=0.2, stratify=detected.COVID_STATUS, random_state=kseed)

rest = post_development_data[~(post_development_data.index.isin(detected.index))].sort_values('record_date')

X_train, Y_train = np.array(train['features'].progress_apply(lambda x: crop_data(pad_data(x))).to_list()), np.array(train['COVID_STATUS'].to_list())
X_valid, Y_valid = np.array(valid['features'].progress_apply(lambda x: crop_data(pad_data(x))).to_list()), np.array(valid['COVID_STATUS'].to_list())
X_test, Y_test = np.array(test['features'].progress_apply(lambda x: crop_data(pad_data(x))).to_list()), np.array(test['COVID_STATUS'].to_list())
X_train_post, Y_train_post = np.array(detected_train['features'].progress_apply(lambda x: crop_data(pad_data(x))).to_list()), np.array(detected_train['COVID_STATUS'].to_list())
X_valid_post, Y_valid_post = np.array(detected_valid['features'].progress_apply(lambda x: crop_data(pad_data(x))).to_list()), np.array(detected_valid['COVID_STATUS'].to_list())
X_test_post, Y_test_post = np.array(rest['features'].progress_apply(lambda x: crop_data(pad_data(x))).to_list()), np.array(rest['COVID_STATUS'].to_list())

clbck = [ModelCheckpoint(model_path / 'model{epoch:02d}.hdf5', save_weights_only=True, verbose=1),
         EarlyStopping(monitor='val_auc', patience=20, verbose=1, restore_best_weights=True, mode="max")]

model = create_al_model(model_path.parent / 'best_model.hdf5')
h = model.fit(X_train_post, Y_train_post, validation_data=(X_valid_post, Y_valid_post), verbose=1, batch_size=16, epochs=100, callbacks=clbck)
