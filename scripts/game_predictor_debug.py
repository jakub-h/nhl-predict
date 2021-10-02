from keras import activations
from sklearn.utils.extmath import softmax
from src.dataset_manager import DatasetManager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow as tf
import plotly.express as px

def prepare_train_val_split():
    dm = DatasetManager("data")
    x, y = dm.get_whole_dataset()
    x = x.dropna(how="any")
    y = pd.get_dummies(y[x.index])

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.75)
    scaler = StandardScaler().fit(x_train)
    x_train_sc = scaler.transform(x_train)
    x_val_sc = scaler.transform(x_val)

    return x_train_sc, x_val_sc, y_train, y_val


def train_NN(train_val_data, epochs, batch_size):
    x_train, x_val, y_train, y_val = train_val_data
    
    model = Sequential()
    model.add(InputLayer(input_shape=x_train.shape[1]))
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation=tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation=tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation=tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation=tf.nn.softmax))
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy', Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]
    )
    history_obj = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val),
        use_multiprocessing=True, workers=16
    )
    return history_obj


def plot_history(history, metric):
    df = pd.DataFrame(history.history)
    fig = px.line(df, y=[metric, f"val_{metric}"])
    fig.show()


if __name__ == "__main__":
    history = train_NN(prepare_train_val_split(), batch_size=8, epochs=50)
    plot_history(history, "accuracy")
