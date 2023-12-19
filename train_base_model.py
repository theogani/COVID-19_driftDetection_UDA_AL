from pathlib import Path
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from numpy.random import seed
from tensorflow.keras.utils import set_random_seed
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import path_to_features, crop_data, pad_data, create_base_model
from utils import get_cambridge_data as read_data

kseed = 1
seed(kseed)
set_random_seed(kseed)
tf.config.experimental.enable_op_determinism()
tqdm.pandas()
model_path = 'Models' / Path('cambridge')
if not model_path.exists():
    model_path.mkdir(parents=True)

data = read_data()
date70 = data.record_date.sort_values().iloc[int(len(data) * 0.7)]
data['features'] = data.progress_apply(path_to_features, axis=1)

development_data = data[data.record_date < date70]
post_development_data = data[data.record_date >= date70]
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
