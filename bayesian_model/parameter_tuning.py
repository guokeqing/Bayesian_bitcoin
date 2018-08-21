from bayesian_model.data_partitioning import Split
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from bayesian_model.performance_evaluation import Evaluation
from bayesian_model.test import calculate_p

class Tuning:
    def __init__(self, train_data):
        self.train_data = train_data

    def kmeans_tuning(self, clusters, state):
        test = Split(self.train_data)
        X_train, X_val = test.split_data()

        param_grid = {"n_clusters": clusters,
                      "random_state": state}
        grid_search = GridSearchCV(KMeans(), param_grid, cv=5)
        for i in range(len(X_train)):
            temp = X_train[i].reshape(-1, 1)
            grid_search.fit(temp)
        print("Best parameters:{}".format(grid_search.best_params_))

    def eval_tuning(self, step, threshold):
        test = Split(self.train_data)
        X_train, X_val = test.split_data()
        p = pd.read_csv(".//p3.csv")
        p = p.values.reshape((1, -1))
        n = 720
        dp = calculate_p()
        bench = np.random.randn(100, 1)
        hold = np.random.randn(1, 100)
        highest_balance = 0

        for i in range(5):
            step_index = np.random.randint(0, len(step))
            threshold_index = np.random.randint(0, len(threshold))
            selected_step = step[step_index]
            selected_threshold = threshold[threshold_index]
            eval = Evaluation(p, n, dp, selected_step, selected_threshold, bench, hold, 100, False)
            balance = eval.visual_account(selected_threshold)[0]
            if highest_balance < balance:
                best_step = selected_step
                best_threshold = selected_threshold
        print("Best step:" + str(best_step) + "     Best threshold: " + str(best_threshold))
        return best_step, best_threshold


if __name__ == '__main__':
    p = pd.read_csv(".//price4.csv")
    p = p.values.reshape((1, -1))
    test = Tuning(p)
    clusters = [98, 100, 103]
    states = [18, 21, 25]
    # test.kmeans_tuning(clusters, states)
    step = [2, 1, 3]
    threshold = [0.01, 0.03, 0.05]
    test.eval_tuning(step, threshold)
