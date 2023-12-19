from sklearn.feature_extraction.image import extract_patches_2d
from librosa import load, power_to_db
from librosa.feature import melspectrogram
from sklearn.metrics import roc_curve, balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from vggish import VGGish
from pathlib import Path
from keras import Input, Model
from keras.backend import clear_session
from keras.layers import Dense, Layer, TimeDistributed
from keras.metrics.metrics import AUC
from keras.optimizers.optimizer_v2.adam import Adam
from keras.src.layers import Rescaling
from keras.src.losses import binary_focal_crossentropy
from keras.models import load_model
import keras.backend as K
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import vggish_params


coswara_csv_path = Path('Coswara-Data') / 'combined_data.csv'
coswara_recordings_path = Path('Coswara-Data') / 'Extracted_data'

cambridge_csv_path = Path('cambridge_data') / 'cambridge' / 'NeurIPs2021-data' / 'task2' / 'data_0426_en_task2.csv'
cambridge_recordings_path = Path('cambridge_data') / 'cambridge' / 'NeurIPs2021-data' / 'covid19_data_0426' / 'New folder' / 'covid19_data_0426'


def get_cambridge_data():
    cambridge = pd.read_csv(cambridge_csv_path, index_col=0)
    cambridge['path'] = cambridge.cough_path.apply(lambda x: cambridge_recordings_path / x.replace('\\', os.sep))
    cambridge.rename({'label': 'COVID_STATUS'}, axis=1, inplace=True)
    cambridge['record_date'] = cambridge.path.apply(lambda x: pd.to_datetime(x.parent.name, format='%Y-%m-%d-%H_%M_%S_%f'))
    return cambridge.drop(cambridge.columns.difference(['path', 'COVID_STATUS', 'record_date', 'uid']), axis=1)


def get_coswara_data():
    cos_data = pd.read_csv(coswara_csv_path)
    cos_data = cos_data[(cos_data.covid_status.isin(['healthy', 'positive_mild', 'positive_moderate', 'positive_asymp'])) & (cos_data.rU != 'y')]

    cos_data['path'] = cos_data.id.apply(lambda x: list(coswara_recordings_path.glob(f'*/{x}/cough-shallow.wav'))[0] if len(list(coswara_recordings_path.glob(f'*/{x}/cough-shallow.wav'))) == 1 else None)

    # some have not any file
    cos_data = cos_data[~cos_data['path'].isna()]

    # some have corrupted files
    cos_data = cos_data[(cos_data.path.apply(lambda x: x.stat().st_size > 500))]

    # some have silent files
    cos_data = cos_data[~(cos_data.progress_apply(lambda x: all(load(x.path, sr=48000)[0] == 0), axis=1))]

    cos_data.rename({'covid_status': 'COVID_STATUS', 'id': 'uid'}, axis=1, inplace=True)

    cos_data.loc[cos_data.COVID_STATUS == 'healthy', 'COVID_STATUS'] = 'n'
    cos_data.loc[cos_data.COVID_STATUS.isin(['positive_mild', 'positive_moderate', 'positive_asymp']), 'COVID_STATUS'] = 'p'

    cos_data['COVID_STATUS'] = (cos_data.COVID_STATUS == 'p') * 1

    cos_data['record_date'] = cos_data.record_date.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))

    return cos_data.drop(cos_data.columns.difference(['path', 'COVID_STATUS', 'record_date', 'uid', 'g']), axis=1, inplace=True)


def path_to_features(x):
    a, sr = load(x.path, sr=vggish_params.SAMPLE_RATE)
    a /= np.abs(a).max()
    S = melspectrogram(y=a, sr=sr, n_fft=int(vggish_params.SAMPLE_RATE * vggish_params.STFT_WINDOW_LENGTH_SECONDS),
                       hop_length=int(vggish_params.SAMPLE_RATE * vggish_params.STFT_HOP_LENGTH_SECONDS),
                       fmin=vggish_params.MEL_MIN_HZ, fmax=vggish_params.MEL_MAX_HZ, n_mels=vggish_params.NUM_MEL_BINS)
    img = power_to_db(S, ref=np.max)
    if np.max(img) != np.min(img):
        img = (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255)
        return img.astype(np.uint8).T


def crop_data(x):
    X = x.copy()
    X = extract_patches_2d(X, (vggish_params.NUM_FRAMES, x.shape[1]))[::vggish_params.NUM_FRAMES // 2]
    return X[..., np.newaxis]


def pad_data(x, length=None):
    if length > x.shape[0]:
        return np.pad(x, ((0, length - x.shape[0]), (0, 0)), mode='wrap')
    else:
        return x[:length, ...]


def create_base_model(input_shape=None):
    clear_session()

    feature_extractor = TimeDistributed(VGGish(include_top=False, load_weights=True))

    inp = Input(shape=input_shape)
    c = tf.cast(inp, tf.float32)
    c = Rescaling(1. / 255.)(c)
    c = feature_extractor(c)
    c = tf.math.reduce_mean(c, axis=1)

    out1 = Dense(1, activation='sigmoid', name='out1')(c)
    m = Model(inp, out1)
    m.compile(optimizer=Adam(lr=1e-4),
              loss=binary_focal_crossentropy, metrics=['acc', AUC()])
    m.summary()
    return m


def evaluate_model(labels, preds, thrs=None):
    if thrs is None:
        _, _, thresholds = roc_curve(labels, preds)
        acc = []
        for thrs in thresholds:
            acc.append(balanced_accuracy_score(labels, (preds > thrs) * 1))

        acc, thrs = max(zip(acc, thresholds), key=lambda x: x[0])
    acc = accuracy_score(labels, (preds > thrs) * 1)
    f = f1_score(labels, (preds > thrs) * 1)
    b = balanced_accuracy_score(labels, (preds > thrs) * 1)
    auc = roc_auc_score(labels, preds)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(labels, (preds > thrs) * 1).ravel()
    print('- - ' * 5 + 'Test Results' + ' - -' * 5)
    print('True positives : ' + str(true_positives))
    print('True negatives : ' + str(true_negatives))
    print('False positives : ' + str(false_positives))
    print('False negatives : ' + str(false_negatives))
    print("Accuracy : {0:}, with threshold : {1:}".format(acc, thrs))
    print("F1 score : {0:}, with threshold : {1:}".format(f, thrs))
    print("Balanced accuracy : {0:}, with threshold : {1:}".format(b, thrs))
    print("ROC_AUC : " + str(auc))
    return thrs


def detect_cusum(x, threshold=1, drift=0):
    x = np.atleast_1d(x).astype('float64')
    gp = np.zeros(x.size)
    t_alarm, t_start = np.array([[], []], dtype=int)
    tap = 0
    amp = np.array([])

    for i in range(1, x.size):
        s = (x[i] - x[i - 1]) / (abs(x[i - 1] + 0.001))
        gp[i] = gp[i - 1] + s - drift
        if gp[i] <= 0:
            gp[i], tap = 0, i
        if gp[i] > threshold:
            t_alarm = np.append(t_alarm, i)
            t_start = np.append(t_start, tap + 1)
            gp[i] = 0

    ind = len(t_start) - 1 - np.unique(np.flip(t_start), return_index=True)[1]
    t_start = t_start[ind]
    t_alarm = t_alarm[ind]

    return t_alarm, t_start, amp, gp


class uda_model(Model):

    def __init__(
            self,
            name="domain_adaptation_model",
            base_model_path=None, **kwargs
    ):
        super(uda_model, self).__init__(name=name, **kwargs)

        self.main = load_model(base_model_path)
        self.main = Model(self.main.inputs[0], self.main.output, name='main')
        self.feature_extractor = Model(self.main.inputs[0], self.main.get_layer('tf.math.reduce_mean').output, name='feature_extractor')
        self.mmd = MMDLayer()

    def call(self, inputs, training=None, **kwargs):
        if training:
            inp_source, inp_target = inputs

            out = self.main(inp_source)

            source_feats = self.feature_extractor(inp_source)
            target_feats = self.feature_extractor(inp_target)

            domain_loss = self.mmd((source_feats, target_feats))

            self.add_loss(domain_loss)

            self.add_metric(domain_loss, name='domain_loss')
        else:
            out = self.main(inputs)
        return out


class MMDLayer(Layer):
    def __init__(self):
        super(MMDLayer, self).__init__()

    def call(self, inputs, **kwargs):
        source_feats, target_feats = inputs
        return compute_mmd(source_feats, target_feats)


def compute_mmd(x, y):
    x_kernel = compute_rbf_kernel(x)
    y_kernel = compute_rbf_kernel(y)
    xy_kernel = compute_rbf_kernel(x, y)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)


def compute_rbf_kernel(X, Y=None, gamma=None):
    """Compute the rbf (gaussian) kernel between X and Y.

        K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows x in X and y in Y.

    Read more in the :ref:`User Guide <rbf_kernel>`.

    Parameters
    ----------
    X : Tensor of shape (n_samples_X, n_features)
        A feature array.

    Y : Tensor of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    gamma : float, default=None
        If None, defaults to 1.0 / n_features.

    Returns
    -------
    kernel_matrix : Tensor of shape (n_samples_X, n_samples_Y)
        The RBF kernel.
    """
    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def euclidean_distances_tf(X, Y):
        # calculate the squared Euclidean distances between each pair of rows in X and Y
        X_expand = tf.expand_dims(X, 1)
        Y_expand = tf.expand_dims(Y, 0)
        return tf.reduce_sum(tf.square(X_expand - Y_expand), axis=2)

    K = euclidean_distances_tf(X, Y)
    K *= -gamma
    K = tf.exp(K)
    return K


def create_uda_model(input_shape=None, base_model_path=None):
    K.clear_session()

    m = uda_model(base_model_path=base_model_path)

    m.build(input_shape)
    m.compile(optimizer=Adam(learning_rate=1e-4),
              loss=binary_focal_crossentropy, metrics=['acc', AUC()])

    m.summary()
    return m


def create_al_model(base_model_path=None):
    K.clear_session()

    m = load_model(base_model_path)

    m.compile(optimizer=Adam(learning_rate=1e-4),
              loss=binary_focal_crossentropy, metrics=['acc', AUC()])

    m.summary()
    return m
