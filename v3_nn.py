from dataset_manager_v3 import DatasetManager
from sklearn.decomposition import PCA
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, InputLayer
import keras
import pandas as pd
import plotly.graph_objects as go


if __name__ == "__main__":
    dm = DatasetManager()
    x_train, x_test, y_train, y_test = dm.get_seasonal_split(2019, 0, 3, 5, True, True, True)
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    pca = PCA(n_components=50)
    scores_train = pca.fit_transform(x_train)
    scores_test = pca.transform(x_test)

    model = Sequential(name="MLP")
    model.add(Dense(128, activation='relu', input_shape=(scores_train.shape[1],), name="hidden_1"))
    model.add(Dropout(rate=0.2, name="dropout_1"))
    model.add(Dense(64, activation='sigmoid', name="hidden_3"))
    model.add(Dropout(rate=0.2, name="dropout_3"))
    model.add(Dense(3, activation='softmax', name='output'))
    model.build()

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])
    n_epochs = 100
    history_obj = model.fit(scores_train, y_train,
                            batch_size=32,
                            epochs=n_epochs,
                            verbose=1,
                            validation_data=(scores_test, y_test))
    history = pd.DataFrame(history_obj.history, index=[i+1 for i in range(n_epochs)])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = history.index,
            y = history["accuracy"],
            mode = "lines+markers",
            name = "Train"
        )
    )
    fig.add_trace(
        go.Scatter(
            x = history.index,
            y = history["val_accuracy"],
            mode = "lines+markers",
            name = "Validation"
        )
    )
    fig.update_layout(
        title = "Model Accuracy",
        xaxis_title = "epochs",
        yaxis_title = "accuracy"
    )
    fig.show()