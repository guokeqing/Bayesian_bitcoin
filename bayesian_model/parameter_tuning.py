import pandas as pd
import numpy as np
from bayesian_model.tuning_test import Test


class Tuning:
    def __init__(self, input_data):
        self.input_data = input_data

    def find_best_params(self, n_list, n_cluster_list, n_effective_list, step_list, threshold_list):
        max_balance = 0
        max_sharpe = 0
        for i in range(10):
            index_n_list = np.random.randint(0, len(n_list))
            index_n_cluster = np.random.randint(0, len(n_cluster_list))
            index_n_effective = np.random.randint(0, len(n_effective_list))
            index_step = np.random.randint(0, len(step_list))
            index_threshold = np.random.randint(0, len(threshold_list))

            temp = Test(self.input_data)
            sharpe_ratio, balance = temp.run_model(n_list[index_n_list], n_cluster_list[index_n_cluster],
                                                   n_effective_list[index_n_effective], step_list[index_step],
                                                   threshold_list[index_threshold])

            if balance > max_balance and sharpe_ratio > max_sharpe:
                max_balance = balance
                max_sharpe = sharpe_ratio
                best_n_list = n_list[index_n_list]
                best_n_cluster = n_cluster_list[index_n_cluster]
                best_n_effective = n_effective_list[index_n_effective]
                best_step = step_list[index_step]
                best_threshold = threshold_list[index_threshold]

        print("Best n_list: " + str(best_n_list) + "    Best number of clusters: " + str(best_n_cluster) +
              "    Best number of effective clusters: " + str(best_n_effective) + "    Best step: " + str(best_step) +
              "    Best threshold: " + str(best_threshold))
        return {'Best n_list': best_n_list, 'Best n_clusters': best_n_cluster, 'Best n_effective': best_n_effective,
                'Best step': best_step, 'Best threshold': best_threshold, 'Balance': balance,
                'Sharpe ratio': sharpe_ratio}

if __name__ == '__main__':
    p = pd.read_csv(".//price4.csv")
    adjust_param = Tuning(p)
    n_list = [[90, 180, 360, 720], [100, 190, 370, 730], [110, 200, 380, 740]]
    n_cluster = [90, 95, 103]
    n_effective = [16, 19, 22]
    step = [1, 2, 3]
    threshold = [0.01, 0.03, 0.05]
    print(adjust_param.find_best_params(n_list, n_cluster, n_effective, step, threshold))
