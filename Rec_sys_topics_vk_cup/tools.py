import numpy as np
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm


class ItemEncoder:
    """
    Класс для кодировки значений в индексы и обратно.
    На вход: список значений в формате pandas.DataFrame или list
    """

    def __init__(self, df, user_col='user_id', item_col='item_id', weight=None):
        self.user_idx = dict(enumerate(df[user_col].unique()))
        self.item_idx = dict(enumerate(df[item_col].unique()))
        self.user_map = {v: k for k, v in self.user_idx.items()}
        self.item_map = {v: k for k, v in self.item_idx.items()}

    def get_users(self, keys, how='idx'):
        """
        Получить значения по индексу
        """
        if how == 'idx':
            return [self.user_map.get(key) for key in keys]
        else:
            return [self.user_idx.get(key) for key in keys]

    def get_items(self, keys, how='idx'):
        """
        Получить значения по индексу
        """
        if how == 'idx':
            return [self.item_map.get(key) for key in keys]
        else:
            return [self.item_idx.get(key) for key in keys]

    def make_csr_data(self, df, user_col='user_id', item_col='item_id', weights=None):
        """
        Создание матрицы user/items
        """
        if weights is None:
            weights = np.ones(len(df), dtype=np.float32)
        else:
            weights = df[weights].astype(np.float32)
        return sp.csr_matrix((weights, (self.get_users(df[user_col]), self.get_items(df[item_col]))), dtype='float32')


def split_data(data, test_size=0.3):
    """Split data by items in order of occurrence. Return (train_data, test_data)"""
    new_data = data.reset_index().groupby('user_id')['index'].unique()
    test_idx = []
    train_idx = []
    for i in tqdm(new_data):
        split = int(i.shape[0] * (1 - test_size))
        train_idx.append(i[:split])
        test_idx.append(i[split:])

    train_idx = np.hstack(train_idx)
    test_idx = np.hstack(test_idx)
    return data.loc[train_idx], data.loc[test_idx]


def compute_metrics(df_true, df_pred, top_N, rank_col='rank'):
    result = {}
    test_recs = df_true.set_index(['user_id', 'item_id']).join(df_pred.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', rank_col])

    test_recs['users_item_count'] = test_recs.groupby(level='user_id')[rank_col].transform(np.size)
    test_recs['reciprocal_rank'] = (1 / test_recs[rank_col]).fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs[rank_col]

    users_count = test_recs.index.get_level_values('user_id').nunique()
    hit_k = f'hit@{top_N}'
    test_recs[hit_k] = test_recs[rank_col] <= top_N
    result[f'Precision@{top_N}'] = (test_recs[hit_k] / top_N).sum() / users_count
    result[f'Recall@{top_N}'] = (test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count

    result[f'MAP@{top_N}'] = (test_recs["cumulative_rank"] / test_recs["users_item_count"]).sum() / users_count
    result[f'MRR'] = test_recs.groupby(level='user_id')['reciprocal_rank'].max().mean()
    return pd.Series(result)


def get_rec_als(model, users, train_matrix, coder, N=100, items=None):
    user_ids = coder.get_users(users)
    if items:
        rec, _ = model.recommend(user_ids, train_matrix, N=N, filter_already_liked_items=True, filter_items=items)
    else:
        rec, _ = model.recommend(user_ids, train_matrix, N=N, filter_already_liked_items=True)
    return user_ids, [coder.get_items(items, how='val') for items in tqdm(rec)]