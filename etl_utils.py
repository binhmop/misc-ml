"""
Beginning input is one csv file, read in as a Pandas DataFrame pdf
ordered by entity and timeStamp
"""
import numpy as np
import pandas as pd
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from sklearn import preprocessing

"""
Schema provides indices of columns in the given PDF
timestamp_column and entity_column are given as string
"""


class DataSet:
    def __init__(self, pdf):
        self.data = pdf  # data is a pd.DataFrame()
        self.nrow = len(pdf)
        self.ncols = len(pdf.columns)

        self.timestamp_column = None  # string name
        self.entity_column = None  # string name
        self.numeric_columns = []
        self.categorical_columns = []
        self.targeted_columns = []
        self.categorical_dict = {}

    def set_timestamp_column(self, timestamp_col):
        self.timestamp_column = timestamp_col

    def set_entity_column(self, entity_col):
        self.entity_column = entity_col

    def set_numeric_columns(self, numeric_cols):
        self.numeric_columns = numeric_cols

    def set_categorical_columns(self, categorical_cols):
        self.categorical_columns = categorical_cols

    def set_targeted_columns(self, targeted_cols):
        self.targeted_columns = targeted_cols

    def generate_categorical_dict(self):
        self.categorical_dict = dict.fromkeys(self.categorical_columns)
        for col_idx in self.categorical_columns:
            n_levels = len(self.data[:, col_idx].unique())
            self.categorical_dict[col_idx] = n_levels

    def set_categorical_dict(self, categorical_dict):
        self.categorical_dict = categorical_dict


def get_label_encodings(pdf, col_idx):
    if np.issubdtype(pdf.iloc[:, col_idx].dtype, np.number):
        return pdf.iloc[:, col_idx]
    else:
        le = preprocessing.LabelEncoder()
        le.fit(pdf.iloc[:, col_idx])
        return pd.Series(le.transform(pdf.iloc[:, col_idx]), index=None)


def set_colnames_with_prefix(df, prefix):
    colnames = []
    ncols = len(df.columns)
    for i in range(ncols):
        colnames.append(prefix + str(i))
    df.columns = colnames


def get_onehot_encodings(pdf, col_idx, col_prefix=None):
    onehot_df = get_label_encodings(pdf, col_idx)
    onehot_df = pd.get_dummies(onehot_df)

    if col_prefix is not None:
        set_colnames_with_prefix(onehot_df, col_prefix)

    return onehot_df


def get_all_onehot_encodings(pdf, col_idc, col_prefix=None):
    l_onehot_df = []
    for i in range(len(col_idc)):
        l_onehot_df.append(get_onehot_encodings(pdf, col_idc[i], col_prefix + str(col_idc[i])))
    return pd.concat(l_onehot_df, axis=1)


def get_feature_embeddings(pdf, col_idx, embedding_size, col_prefix=None, num_levels=None):
    col_series = get_label_encodings(pdf, col_idx)

    if num_levels is None:
        num_levels = len(col_series.unique())
    emb_model = Sequential()
    emb_model.add(Embedding(num_levels, embedding_size))
    emb_model.compile('rmsprop', 'mse')
    embs = emb_model.predict(np.arange(num_levels)).reshape(num_levels, embedding_size)  # shape (n_levels, emb_size)

    l_emb_df = []
    for i in range(len(col_series)):
        l_emb_df.append(embs[col_series.iloc[i],])

    emb_df = pd.DataFrame(np.array(l_emb_df), index=None)
    if col_prefix is not None:
        set_colnames_with_prefix(emb_df, col_prefix)

    return emb_df


def get_all_feature_embedings(pdf, col_idc, embedding_size, categorical_dict, col_prefix=None):
    l_col_embs = []
    for col_idx in col_idc:
        if col_idx in categorical_dict:
            l_col_embs.append(get_feature_embeddings(pdf, col_idx, embedding_size, col_prefix + str(col_idx),
                                                     categorical_dict[col_idx]))
        else:
            l_col_embs.append(get_feature_embeddings(pdf, col_idx, embedding_size, col_prefix + str(col_idx)))
    return pd.concat(l_col_embs, axis=1)


def encode_categorical_columns(dataset, embedding_size=12, num_level_threshold=12):
    dataset.generate_categorical_dict()
    pdf = dataset.data
    onehot_idc = []
    emb_idc = []
    for idx in dataset.categorical_columns:
        if dataset.categorical_dict[idx] > num_level_threshold:
            emb_idc.append(idx)
        else:
            onehot_idc.append(idx)
    onehot_df = get_all_onehot_encodings(pdf, onehot_idc, col_prefixes='onehot_feat_')
    emb_df = get_all_feature_embedings(pdf, emb_idc, embedding_size, dataset.categorical_dict, col_prefix='emb_feat_')
    return onehot_df, emb_df


def get_feature_scalers(dataset, scaler=preprocessing.MinMaxScaler(), test_size=0.25):
    """
    scaler is a sklearn.preprocessing scaler, e.g. MinMaxScaler, MaxAbsScaler
    """
    pdf = dataset.data
    y_scaler, x_scaler = None, None

    if dataset.entity_column is None:
        test_size = round(len(pdf) * test_size) if test_size <= 1 else test_size
        if len(dataset.numeric_columns) > 0:
            numeric_df_train = pdf.iloc[0:-test_size, dataset.numeric_columns]
            x_scaler = scaler.fit(numeric_df_train)

        y_df_train = pdf.iloc[0:-test_size, dataset.targeted_columns]
        y_scaler = scaler.fit(y_df_train)

    else:
        x_df_train = []
        y_df_train = []
        entity_col = dataset.entity_column
        l_entity_values = pdf[entity_col].unique()
        for entity_id in l_entity_values:
            cur_pdf = pdf[pdf[entity_col] == entity_id]
            test_size = round(len(cur_pdf) * test_size) if test_size <= 1 else test_size
            if len(dataset.numeric_columns) > 0:
                x_df_train.append(cur_pdf.iloc[0:-test_size, dataset.numeric_columns])
            y_df_train.append(cur_pdf.iloc[0:-test_size, dataset.targeted_columns])

        if len(x_df_train) > 0:
            x_df_train = pd.concat(x_df_train, axis=0)
            x_scaler = scaler.fit(x_df_train)

        y_df_train = pd.concat(y_df_train, axis=0)
        y_scaler = scaler.fit(y_df_train)

    return y_scaler, x_scaler


def normalize_data(dataset, y_scaler, x_scaler=None):
    pdf = dataset.data
    scaled_numeric_df = None if x_scaler is None else pd.DataFrame(
        x_scaler.transform(pdf.iloc[:, dataset.numeric_columns]), index=None)
    scaled_targeted_df = pd.DataFrame(y_scaler.transform(pdf.iloc[:, dataset.targeted_columns]), index=None)
    return scaled_numeric_df, scaled_targeted_df


def generate_seqs(data, ycols, n_prev=30, n_next=1):
    xs, ys = [], []
    for i in range(len(data) - n_prev - n_next):
        xs.append(data.iloc[i:i + n_prev, :].as_matrix())
        ys.append(data.iloc[i + n_prev: i + n_prev + n_next, ycols].as_matrix())
    idx = len(data) - n_prev - n_next + 1
    xs.append(data.iloc[idx: idx + n_prev, :].as_matrix())
    ys.append(data.iloc[-n_next:, ycols].as_matrix())

    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


def train_val_split(data, ycols, seq_len=1, n_next=1, test_size=0.25):
    """
    data is pd.DataFrame() and normalized
    seq_len > 1: generate sequences with length = seq_len for RNN, default value = 1 for use in FFN
    n_next > 1: predict n_next timesteps, default value = 1 to predict immediate next timestep
    """
    xcols = [idx for idx in range(len(data)) if idx not in ycols]
    test_size = round(len(data) * test_size) if test_size <= 1 else test_size

    xs, ys = generate_seqs(data, ycols, seq_len, n_next) if seq_len > 1 \
                 else np.array(data[xcols]), np.array(data[ycols])
    xtrain_scaled, ytrain_scaled = xs[0:-test_size], ys[0:-test_size]
    xval_scaled, yval_scaled = xs[-test_size:], ys[-test_size:]
    return xtrain_scaled, ytrain_scaled, xval_scaled, yval_scaled


def train_val_split_all(data, entity_col, ycols, n_timesteps=1, n_next=1, test_size=0.25):
    l_xt_scaled, l_yt_scaled, l_xv_scaled, l_yv_scaled = [], [], [], []

    l_entity_values = data[entity_col].unique()
    for entity_id in l_entity_values:
        cur_data = data[data[entity_col] == entity_id]
        cur_data.drop([entity_col], axis=1, inplace=True)
        xt_scaled, yt_scaled, xv_scaled, yv_scaled = train_val_split(cur_data, ycols, n_timesteps,
                                                                     n_next, test_size)
        l_xt_scaled.append(xt_scaled)
        l_yt_scaled.append(yt_scaled)
        l_xv_scaled.append(xv_scaled)
        l_yv_scaled.append(yv_scaled)

    xtrain_scaled = np.vstack(l_xt_scaled)
    ytrain_scaled = np.vstack(l_yt_scaled)
    xval_scaled = np.vstack(l_xv_scaled)
    yval_scaled = np.vstack(l_yv_scaled)
    return xtrain_scaled, ytrain_scaled, xval_scaled, yval_scaled


def prepare_data(dataset, test_size=0.25, embedding_size=12, num_level_threshold=12, n_timesteps=1, n_next=1):
    pdf = dataset.data
    entity_df = pdf.loc[:, [dataset.entity_column]]

    # encoding data
    onehot_df, emb_df = encode_categorical_columns(dataset, embedding_size, num_level_threshold)
    # scaling data
    y_scaler, x_scaler = get_feature_scalers(dataset, test_size)
    scaled_numeric_df, scaled_targeted_df = normalize_data(dataset, y_scaler, x_scaler)
    final_df = pd.concat([entity_df, onehot_df, emb_df, scaled_numeric_df, scaled_targeted_df], axis=1)

    # split data to train and test
    ycols_idc = list(range(final_df.shape[1]))[-scaled_numeric_df.shape[1]:]
    xtrain_scaled, ytrain_scaled, xval_scaled, yval_scaled = \
        train_val_split_all(final_df, dataset.entity_column, ycols_idc, n_timesteps, n_next, test_size)
    return xtrain_scaled, ytrain_scaled, xval_scaled, yval_scaled
