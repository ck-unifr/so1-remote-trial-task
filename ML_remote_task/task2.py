import pandas as pd
import numpy as np

import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import StandardScaler


TRAIN_FILE = 'train.csv'


def data_preparation(data_user, weeks, product_ids, nb_prev_months=3):
    """
    create x_train, y_train, x_test
    :param data_user: a pandas dataframe contains the purchase data
    :param weeks: a list which contains the indices of weeks
    :param product_ids: a list which contains the product id
    :param nb_prev_months: a integer. In order to predict the probabilities of the products to purchase, we need the data of n previous months.
    :return:
    """

    # products purchased for n previous months
    X_train = []
    # products purchased in the current months
    Y_train = []

    weeks.sort()
    for week in range(nb_prev_months, len(weeks)):
        # create targets
        current_product_purchased = [0] * product_ids
        for i in data_user[data_user['t'] == week]['j'].values:
            # print(i)
            current_product_purchased[i] += 1
        Y_train.append(current_product_purchased)

        # print('....')
        # print((data_user[data_user['t'] == week]['j'].values))
        # Y_train.append((data_user[data_user['t'] == week]['j'].values))


        # create features
        # print('length')
        # print(len(product_ids))
        total_product_purchased = [0.0] * product_ids
        total_product_advertised = [0] * product_ids

        for i in range(nb_prev_months):
            products_purchased = data_user[data_user['t'] == (week-i-1)]
            for i, p_id in enumerate(products_purchased['j'].values):
                total_product_purchased[p_id] += (products_purchased['price'].values)[i]
                total_product_advertised[p_id] += (products_purchased['advertised'].values)[i]
        # print(len(total_product_purchased))
        # print(len(total_product_advertised))
        X_train.append(np.concatenate([total_product_purchased,total_product_advertised]))

    X_test = []
    total_product_purchased = [0.0] * product_ids
    total_product_advertised = [0] * product_ids
    for i in range(nb_prev_months):
        products_purchased = data_user[data_user['t'] == (weeks[-1 - i])]
        for i, p_id in enumerate(products_purchased['j'].values):
            total_product_purchased[p_id] += (products_purchased['price'].values)[i]
            total_product_advertised[p_id] += (products_purchased['advertised'].values)[i]
    X_test.append(np.concatenate([total_product_purchased, total_product_advertised]))


    print('shape')
    print(np.array(X_train).shape)
    print(np.array(Y_train).shape)
    print(np.array(X_test).shape)
    print('-----')

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return np.array(X_train), np.array(Y_train), np.array(X_test)


def model_training(X_train, Y_train, num_classes, model_type='mlp', epochs=10, batch_size=128):
    model = None
    if model_type == 'mlp':
        # model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        # out = ks.layers.Dense(256, activation='relu')(model_in)
        # # out = ks.layers.Dense(128, activation='relu')(out)
        # out = ks.layers.Dense(64, activation='relu')(out)
        # out = ks.layers.Dense(len(product_ids), activation="sigmoid")(out)
        # model = ks.Model(model_in, out)
        # lr_init = 1e-3
        # model.compile(loss='binary_crossentropy', optimizer=ks.optimizers.Adam(lr=lr_init))
        lr_init = 1e-3
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=ks.optimizers.Adam(lr=lr_init))

        y_train = Y_train
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

        model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # print('Total CV score is {}'.format(np.mean(scores)))
    return model

def prediction(model, X_test, model_type='mlp'):
    Y_test = None
    if model_type == 'mlp':
        Y_test = model.predict(X_test)

    print(Y_test.shape)
    print(Y_test)

    return Y_test




data = pd.read_csv(TRAIN_FILE)

# print(data.head())

user_id_list = data['i'].unique()

# print(users)

product_id_list = data['j'].unique()

# print(products)

weeks = data['t'].unique()

# print(weeks)

# sub_data = data[data['i'] == 4]
# print(sub_data)

for user_id in user_id_list:
    data_user = data[data['i'] == user_id]
    # print(data_user.shape)

data_user = data[data['i'] == 2]



X_train, Y_train, X_test = data_preparation(data_user, weeks, product_id_list)

model = model_training(X_train, Y_train, len(product_id_list))

print('x test shape')
print(X_test.shape)

prediction(model, X_test,)


for product_id in product_id_list:
    prices = data[data['j'] == product_id]['price']
    # print(prices.unique())



