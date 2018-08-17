class Split:
    def __init__(self, input_data):
        self.input_data = input_data

    def split_data(self):
        train_data = []
        test_data = []

        for i in range(10):
            if len(self.input_data) == 1:
                step = round(len(self.input_data[0]) / 10)
                index = i * step
                train_data.append(self.input_data[0, index: index + round(step * (4 / 5))])
                test_data.append(self.input_data[0, index + round(step * (4 / 5)): index + step])
            else:
                step = round(len(self.input_data) / 10)
                index = i * step
                train_data.append(self.input_data[index: index + round(step * (4/5))])
                test_data.append(self.input_data[index + round(step * (4/5)): index + step])

        return train_data, test_data

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.cluster import KMeans
    from bayesian_model.data_processor import ProcessData

    p = pd.read_csv(".//price4.csv")
    p = p.values.reshape((1, -1))
    p2 = np.arange(1, len(p[0]) + 1)

    test = Split(p)
    X_train, X_val = test.split_data()
    test2 = Split(p2)
    y_train, y_val = test2.split_data()

    param_grid = {"n_clusters": [98, 101, 105],
                  "random_state": [25, 19, 31]}
    grid_search = GridSearchCV(KMeans(), param_grid, cv=5)
    for i in range(len(X_train)):
        temp = X_train[i].reshape(-1, 1)
        temp2 = y_train[i].reshape(-1, 1)
        grid_search.fit(temp, temp2)
    print("Best parameters:{}".format(grid_search.best_params_))


