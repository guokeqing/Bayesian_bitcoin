from bayesian_model.data_processor import ProcessData
from bayesian_model.bayesian_regression import Prediction
from bayesian_model.performance_evaluation import Evaluation
from bayesian_model.index import Calculate_index
import numpy as np


class Test:
    def __init__(self, input_data):
        self.input_data = input_data

    def run_model(self, n, n_cluster, n_effective, step, threshold):
        data = self.input_data
        data = data.values.reshape(1, -1)
        index = round(len(data[0]) / 3)
        p1 = data[0, 0: index]
        p2 = data[0, index: 2 * index]
        p_eval = data[0, 2 * index:]

        data_pro = ProcessData(p1, n, n_cluster, n_effective)
        effective = data_pro.select_effective_clusters()

        test_model = Prediction(effective, p2, n, p_eval)
        p = test_model.predict_delta_p()

        bench = np.random.randn(100, 1)
        hold = np.random.randn(1, 100)

        eval_result = Evaluation(p_eval, max(n), p, step, threshold, bench, hold, 100, True, 5000, 5000, 4)
        returns = eval_result.periodic_return()[0]
        market = eval_result.periodic_return()[1]
        temp = Calculate_index(returns, market, 0.05, 0.04, 1, 500, 4)
        sharpe = temp.sharpe_ratio()
        return sharpe, eval_result.visual_account(threshold)[0], eval_result.visual_account(threshold)[2]
