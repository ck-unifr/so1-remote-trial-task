#------------------------------------
#
# SO1 Remote Trial Task
# Task 2: Predict Consumer Purchases
#
# Author: Kai Chen
# Date: Mar, 2018
#
#------------------------------------

import pandas as pd
import numpy as np
import math

import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import time

TRAIN_FILE = 'train.csv'
PRODUCT_DISCOUNT_FILE = 'promotion_schedule.csv'


def get_product_prom(product_discount_file, product_id_list):
    product_dis_dict = {}
    product_adver_dict = {}

    for id in product_id_list:
        product_dis_dict[id] = 0.0
        product_adver_dict[id] = 0

    data = pd.read_csv(product_discount_file)

    product_dis = data['discount']
    product_adver = data['advertised']
    product_id = data['j']

    for i, id in enumerate(product_id):
        product_dis_dict[id] = product_dis[i]
        product_adver_dict[id] = product_adver[i]

    return product_dis_dict, product_adver_dict

# product_dis_dict, product_adver_dict = get_product_prom(PRODUCT_DISCOUNT_FILE, [])


def get_product_price_list(product_purchase_file, product_id_list):
    product_price_dict = {}
    for product_id in product_id_list:
        product_price_dict[product_id] = [0.0]

    data = pd.read_csv(product_purchase_file)

    product_id = data['j'].unique()
    weeks = data['t'].unique()

    for id in product_id:
        product = data[data['j']==id]
        product_price_list = []
        for week in weeks:
            prices = product[product['t'] == week]['price'].values
            if len(prices) > 0:
                price = prices.mean()
                if not math.isnan(price):
                    product_price_list.append(price)
        product_price_dict[id] = product_price_list

    return product_price_dict

# product_price_dict = get_product_price_list(TRAIN_FILE, [])


# TODO: estimate the product price in the future
def get_product_price_next_month(product_price_dict):
    product_price_next_month_dict = {}
    for id, values in product_price_dict.items():
        product_price_next_month_dict[id] = np.mean(values)
    return product_price_next_month_dict

# product_price_next_month = get_product_price_next_month(product_price)


#TODO: try different features
def prepare_data(data_user, weeks, product_ids, product_discount_dict,
                     product_price_next_month_dict, product_adver_next_month_dict, nb_prev_months=2):
    """
    create x_train, y_train, x_test
    :param data_user: a pandas dataframe contains the purchase data
    :param weeks: a list which contains the indices of weeks
    :param product_ids: a list which contains the product id
    :param nb_prev_months: a integer. In order to predict the probabilities of the products to purchase, we need the data of n previous months.
    :param product_price_next_month: a list contains the products prices
    :param product_adver_next_month: a one-hot-encoded list contains the information of product advertisement (e.g., 0 or 1)
    :return:
    """

    # products purchased for n previous months
    X_train = []
    # products purchased in the current months
    Y_train = []

    weeks.sort()
    for week in range(nb_prev_months, len(weeks)):

        #---------------
        # create targets
        current_product_purchased = [0] * product_ids
        for j in data_user[data_user['t'] == week]['j'].values:
            current_product_purchased[j] += 1
        Y_train.append(current_product_purchased)

        # ---------------
        # create features
        products_purchased = data_user[data_user['t'] == week]

        product_price_cur = [0.0] * product_ids
        product_adver_cur = [0] * product_ids
        for p_id in products_purchased['j'].values:
            product_price_cur[p_id] = (products_purchased['price'].values).mean()
            product_adver_cur[p_id] = int((products_purchased['advertised'].values).mean())

        product_price_prev = [0.0] * product_ids
        product_adver_prev = [0] * product_ids
        for i in range(nb_prev_months):
            products_purchased = data_user[data_user['t'] == (week-i-1)]
            for i, p_id in enumerate(products_purchased['j'].values):
                product_price_prev[p_id] += (products_purchased['price'].values)[i]
                product_adver_prev[p_id] += (products_purchased['advertised'].values)[i]

        X_train.append(np.concatenate([product_price_cur, product_adver_cur, product_price_prev, product_adver_prev]))

    #------------
    # create test set (purchase of next month)
    X_test = []

    product_price_cur = [0.0] * product_ids
    product_adver_cur = [0] * product_ids
    for id, value in product_price_next_month_dict.items():
        product_price_cur[id] = value * (1 - product_discount_dict[p_id])
    for id, value in product_adver_next_month_dict.items():
        product_adver_cur[id] = value

    product_price_prev = [0.0] * product_ids
    product_adver_prev = [0] * product_ids
    for i in range(nb_prev_months):
        products_purchased = data_user[data_user['t'] == (weeks[-1 - i])]
        for i, p_id in enumerate(products_purchased['j'].values):
            product_price_prev[p_id] += (products_purchased['price'].values)[i]
            product_adver_prev[p_id] += (products_purchased['advertised'].values)[i]

    X_test.append(np.concatenate([product_price_cur, product_adver_cur, product_price_prev, product_adver_prev]))

    # print('--------------------------')
    # print('data sets shape')
    # print('x train')
    # print(np.array(X_train).shape)
    # print('y train')
    # print(np.array(Y_train).shape)
    # print('x test')
    # print(np.array(X_test).shape)
    # print('--------------------------')

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return np.array(X_train), np.array(Y_train), np.array(X_test)


# TODO: try different NN architectures
def train_mlp(X_train, Y_train, num_classes, epochs=10, batch_size=64, lr_init = 1e-3):
    # model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
    # out = ks.layers.Dense(256, activation='relu')(model_in)
    # # out = ks.layers.Dense(128, activation='relu')(out)
    # out = ks.layers.Dense(64, activation='relu')(out)
    # out = ks.layers.Dense(len(product_ids), activation="sigmoid")(out)
    # model = ks.Model(model_in, out)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=ks.optimizers.Adam(lr=lr_init), metrics=['accuracy'])

    # y_train = []
    # for y in Y_train:
    #     i = 0
    #     for a, y_i in enumerate(y):
    #         if y_i != 0: i = a
    #     y_train.append(i)
    # y_train = np.reshape(y_train, (X_train.shape[0], 1))
    # print('y_train shape')
    # print(y_train.shape)
    # y_train = ks.utils.to_categorical(y_train, num_classes)
    # print(y_train.shape)

    model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    return model

def predict(model, X_test, model_type='mlp'):
    Y_test = None
    if model_type == 'mlp':
        Y_test = model.predict(X_test)

    # print(Y_test.shape)
    # print(Y_test)
    return np.reshape(Y_test, (-1, 1))


# TODO: k-folds cross validation for hyperparameter tuning
def evaluate(X_train, Y_train, num_classes):
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)
    model = train_mlp(x_train, y_train, num_classes, epochs=10, batch_size=64, lr_init=1e-3)
    y_pred = model.predict(x_val)
    score = roc_auc_score(y_val, y_pred)
    return score


if __name__ == "__main__":
    data = pd.read_csv(TRAIN_FILE)
    # print(data.head())

    # get user id list
    user_id_list = data['i'].unique()
    print('number of users: {}'.format(len(user_id_list)))
    # print(users)

    # get product id list
    product_id_list = data['j'].unique()
    print('number of products: {}'.format(len(product_id_list)))
    # print(products)

    # get week list
    weeks = data['t'].unique()
    print('number of weeks: {}'.format(len(weeks)))
    # print(weeks)

    # check product price changement
    print('product price')
    for product_id in product_id_list:
        prices = data[data['j'] == product_id]['price']
        print('product id: {}'.format(product_id))
        prices = prices.unique()
        print('mean (price): {}'.format(np.mean(prices)))
        print('standard deviation (price): {}'.format(np.std(prices)))

    # get product discount dictionary
    # get product advertisement dictionary
    product_dis_dict, product_adver_dict = get_product_prom(PRODUCT_DISCOUNT_FILE, product_id_list)

    # get product price list
    product_price_list = get_product_price_list(TRAIN_FILE, product_id_list)

    # get the product list of next month
    product_price_next_month = get_product_price_next_month(product_price_list)

    # predict consumer purchases
    nb_prev_months = 1
    epochs = 1
    batch_size = 128
    prediction_i = []
    prediction_j = []
    prediction_prob = []
    # user_id_list = user_id_list[:2]
    tic = time.clock()
    for n_user, user_id in enumerate(user_id_list):
        data_user = data[data['i'] == user_id]

        X_train, Y_train, X_test = prepare_data(data_user, weeks, product_id_list, product_dis_dict,
                                                product_price_next_month, product_adver_dict,
                                                nb_prev_months=nb_prev_months)

        # evaluate the model
        # score = evaluate(X_train=X_train, Y_train=Y_train, num_classes=len(product_id_list))
        # print('score (auc): {}'.format(score))

        model = train_mlp(X_train=X_train, Y_train=Y_train, num_classes=len(product_id_list),
                          epochs=epochs, batch_size=batch_size)

        Y_test = predict(model, X_test)

        if ((n_user+1) % len(user_id_list) == 50):
            toc = time.clock()
            time_used = (toc-tic)/60.0
            time_required = (time_used / (n_user+1))*(len(user_id_list) - n_user -1)
            print('{}/{}'.format(str(n_user + 1), len(user_id_list)))
            print('time used {} mins.'.format(time_used))
            print('time required {} mins.'.format(time_required))

        for j, y_test in enumerate(Y_test):
            prediction_i.append(user_id)
            prediction_j.append(j)
            prediction_prob.append(y_test[0])

    prediction_df = pd.DataFrame({'i':prediction_i,
                                  'j':prediction_j,
                                  'prediction':prediction_prob})
    prediction_file = 'prediction-mlp-[prev-month]_{}-[epoch]_{}-[batch-size]_{}.csv'.format(str(nb_prev_months),
                                                                                             str(epochs), str(batch_size))
    prediction_df.to_csv(prediction_file, index=False)
    print('save prediction to {}'.format(prediction_file))


# -------------------
# Future work
#
# 1. Feature engineering
# 2. Modeling the time dependence of product purchase with recurrent neural networks, e.g., GRU, LSTM
# 3. Speed up the training process with Spark
# 4. Train one model for all the users? We could consider user id as a feature?
# 5. k-folds cross validation for hyperparameter tuning
#


