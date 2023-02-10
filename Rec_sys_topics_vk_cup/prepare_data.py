import pandas as pd
from tools import split_data


def make_data():
    data = pd.read_parquet('files/vk_data/train.parquet.gzip')
    #data = data.loc[data['timespent'] > 0]
    train_data, test_data = split_data(data, test_size=0.2)

    train_data.to_parquet('files/global_train.parquet.gzip', compression='gzip')
    test_data.to_parquet('files/global_test.parquet.gzip', compression='gzip')


def make_features():
    path_item = 'files/item_features.parquet.gzip'
    path_user = 'files/user_features.parquet.gzip'

    df = pd.read_parquet('files/vk_data/train.parquet.gzip')
    df_items = pd.read_parquet('files/vk_data/items_meta.parquet.gzip')

    item_features = pd.crosstab(df['item_id'], df['reaction'])
    item_features_1 = df.groupby('item_id')['timespent'].agg(['sum', 'mean']).astype('float32')
    item_features_1.columns = ['sum_time', 'mean_time']
    item_features.columns = ['likes', 'no_likes', 'dislikes']
    item_features = pd.concat([item_features, item_features_1], axis=1)
    item_features = item_features.merge(df_items, on=['item_id'])

    user_features = df.groupby('user_id')[['timespent', 'reaction']].agg(['sum', 'mean']).astype('float32')
    user_features.columns = ['timespent_sum', 'time_spent_mean', 'reaction_sum', 'reaction_mean']
    user_features.loc[:, 'cou_posts'] = df.groupby('user_id')['item_id'].nunique().astype('int16')

    item_features['source_id'] = item_features['source_id'].astype('category')
    item_features['item_id'] = item_features['item_id'].astype('int32')
    item_features['likes'] = item_features['likes'].astype('int32')
    item_features['no_likes'] = item_features['no_likes'].astype('int32')
    item_features['dislikes'] = item_features['dislikes'].astype('int32')

    item_features.to_parquet(path_item)
    user_features.to_parquet(path_user)


if __name__ == '__main__':
    print('make_data...')
    make_data()
    print('make_features....')
    make_features()
