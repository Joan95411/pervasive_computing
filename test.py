import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import f1_score



def generate_model(num_out=3, input_shape=(100, 52, 2)):
    inputs = layers.Input(shape=input_shape)  # this is your input
    z = layers.Flatten()(inputs)

    ''' Under this is your neural network playing field '''
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(100, 'relu')(z)
    ''' Above this is your neural network playing field '''

    output = layers.Dense(num_out, activation="softmax")(z)

    model = models.Model(inputs=inputs, outputs=output)
    return model


def normalize(arr):
    return (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))


def normalize_input(data):
    norm = normalize(data)
    norm = norm - np.mean(norm)
    return norm


def create_filename(n, m, p, l, a, f_base="data/spotlight/processed"):
    return f"{f_base}/{n}-{m}_{l}_{a}_{p}.npy"


def load_csi(n, m, p, l, a, data_folder, norm=True, diff=False):
    f_name = create_filename(n=n, m=m, p=p, l=l, a=a, f_base=data_folder)
    csi = np.load(f_name)
    return csi


def get_location_data(node_dict, n, m, L, t, subset=1.0):
    relevant_branch = node_dict[n][m]
    x, y = [], []
    for l, a_dict in relevant_branch.items():
        for a, e in a_dict.items():  # e containts the data [x] and index of labels for train, test val
            csi = e['x']
            if len(csi.shape) == 4:  # make sure it has enough elements
                t_csi = csi[e[t]]
                x.append(t_csi)
                y.append([L.index(l)] * len(csi[e[t]]))
    return np.concatenate(x, axis=0) if len(x) > 0 else [], np.concatenate(y, axis=0) if len(y) > 0 else []




def generate_node_dict(data_folder, N, L, A, p, split_train_percentage, split_val_percentage, diff=False, mix=True,
                       norm=True):
    node_data: dict = {}

    combinations = [(n, m, l, a) for a in A for l in L for n in N for m in N if n != m]
    to_be_deleted = []
    for n, m, l, a in combinations:
        if n not in node_data.keys(): node_data[n] = {}
        if m not in node_data[n].keys(): node_data[n][m] = {}
        if l not in node_data[n][m].keys(): node_data[n][m][l] = {}
        if a not in node_data[n][m][l].keys(): node_data[n][m][l][a] = {}

        csi = load_csi(n=n, m=m, l=l, p=p, a=a, data_folder=data_folder)
        if len(csi.shape) < 4 or csi.shape[0] == 0:
            to_be_deleted.append((n, m))
        else:
            if diff:
                csi = np.diff(csi, n=1, axis=diff)
            if norm:
                csi = normalize_input(csi)

            test_percentage = split_train_percentage + split_val_percentage
            index_range_end = int(test_percentage * csi.shape[0])
            num_train_elem = int(csi.shape[0] * split_train_percentage)
            indices = np.arange(0, index_range_end)
            train_idx = np.asarray(indices[:num_train_elem], dtype=int)
            val_idx = np.asarray(indices[num_train_elem:], dtype=int)
            test_idx = np.arange(index_range_end, csi.shape[0])

            if (split_val_percentage > 0 and len(val_idx) == 0) or len(train_idx) == 0:
                print(f"\tIncomplete data for {m}->{n} for {l}....")
                to_be_deleted.append((n, m))
            else:
                node_data[n][m][l][a] = {'x': csi, 'train': train_idx, 'val': val_idx, 'test': test_idx}

    for n, m in list(set(to_be_deleted)):
        del node_data[n][m]

    return node_data


def train_test():
    N = [10, 11, 12, 13, 14, 15,
         16]  # These are all the devices involved for sensing,  you can eventually pick well-performing subsets
    p = 1
    L = ['desk', 'bed', 'bathroom']  # Locations
    A = ['sitstand', 'nothing', 'work', 'aggitated']
    train_percentage = 0.4
    validation_percentage = 0.2
    info_dict = generate_node_dict('data', N, L, A, p, train_percentage, validation_percentage)
    x_train, y_train = get_location_data(info_dict, n=N[0], m=N[1], L=L,
                                         t="train")  # n = transmitter, m = receiver, n != m
    x_val, y_val = get_location_data(info_dict, n=N[0], m=N[1], L=L, t="val")
    x_test, y_test = get_location_data(info_dict, n=N[0], m=N[1], L=L, t="test")
    # Shuffle the test data indices
    # test_indices = list(range(len(x_test)))
    # random.shuffle(test_indices)
    # x_test_shuffled = x_test[test_indices]
    # y_test_shuffled = y_test[test_indices]

    model: models.Model = generate_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=50, batch_size=4, validation_data=(x_val, y_val))

    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average='weighted')
    print(f1)

    return [len(x_test), y_pred]
    # y_pred = model.predict(x_test_shuffled)
    # f1 = f1_score(y_test_shuffled, np.argmax(y_pred, axis=1), average='weighted')
    # print(f1)

    #return [len(x_test_shuffled), y_pred]


