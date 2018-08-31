from bayesian_model.data_processor import ProcessData
from bayesian_model.bayesian_regression import Prediction
from bayesian_model.performance_evaluation import Evaluation
from bayesian_model.index import Calculate_index
import numpy as np


class Test:
    def __init__(self, input_data):
        self.input_data = input_data

    def run_model(self, n, n_cluster, n_effective, step, threshold, is_ultimate):
        """
        :param n: 一组设定时间序列长度的数组
        :param n_cluster: int, 聚类个数
        :param n_effective: 有效聚类个数
        :param step: 步长
        :param threshold: 阈值
        :param is_ultimate: 波尔数，是否运行最终结果
        :return:夏普指数，账户余额
        """
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
        drawdown = eval_result.calculate_max_drawdown()
        #returns = eval_result.periodic_return()[0]
        #market = eval_result.periodic_return()[1]
        #temp = Calculate_index(returns, market, 0.05, 0.04, 1, 500, 4)
        # sharpe = temp.sharpe_ratio()

        if is_ultimate:
            rate = eval_result.correct_rate()
            print("Correct rate:", rate)
            eval_result.plot_price_and_profit()

        return drawdown, eval_result.visual_account(threshold)[0], eval_result.visual_account(threshold)[2]
