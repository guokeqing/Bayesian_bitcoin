import pandas as pd
import numpy as np
from bayesian_model.tuning_test import Test


class Tuning:
    def __init__(self, input_data):
        self.input_data = input_data

    def find_best_params(self, n_list, n_cluster_list, n_effective_list, step_list, threshold_list):
        """
        :param n_list: 一个嵌套数列，每一个元素都是一个长度为4的设置时间序列长度的数列
        :param n_cluster_list: 一个数列，每一个元素为一个聚类个数值
        :param n_effective_list: 数列，每个元素为一个有效聚类个数值
        :param step_list: 数列，每个元素为一个步长值
        :param threshold_list: 数列，每个元素为一个阈值
        :return: a dictionary that contains the best parameters found
        随机从输入的参数数列中选取参数值，把选出的参数组合代入模型进行数据预测并评估模型表现。
        选择虚拟账户余额最高及夏普比率最高，或者两个评估指标中其中一个相对于目前最大值的增长较多的参数组合。
        """
        max_balance = 0
        min_drawdown = 100
        for i in range(10):
            index_n_list = np.random.randint(0, len(n_list))
            index_n_cluster = np.random.randint(0, len(n_cluster_list))
            index_n_effective = np.random.randint(0, len(n_effective_list))
            index_step = np.random.randint(0, len(step_list))
            index_threshold = np.random.randint(0, len(threshold_list))

            temp_test = Test(self.input_data)
            drawdown, balance, profit = temp_test.run_model(n_list[index_n_list], n_cluster_list[index_n_cluster],
                                                                n_effective_list[index_n_effective],
                                                                step_list[index_step],
                                                                threshold_list[index_threshold], False)

            if balance > max_balance:
                max_balance = balance
                min_drawdown = drawdown
                best_n_list = n_list[index_n_list]
                best_n_cluster = n_cluster_list[index_n_cluster]
                best_n_effective = n_effective_list[index_n_effective]
                best_step = step_list[index_step]
                best_threshold = threshold_list[index_threshold]

        return {'Best n_list': best_n_list, 'Best n_clusters': best_n_cluster, 'Best n_effective': best_n_effective,
                'Best step': best_step, 'Best threshold': best_threshold, 'Balance': max_balance,
                'Max drawdown': min_drawdown}


if __name__ == '__main__':
    p = pd.read_csv(".//price4.csv")
    adjust_param = Tuning(p)
    n_list = [[90, 180, 360, 720], [100, 190, 370, 730], [95, 185, 365, 725], [110, 200, 380, 740]]
    n_cluster = [90, 95, 98, 100, 101]
    n_effective = [13, 15, 16, 19, 22]
    step = [1, 2, 3]
    threshold = [0.01, 0.001, 0.007]
    temp = adjust_param.find_best_params(n_list, n_cluster, n_effective, step, threshold)
    print(temp)

    p2 = pd.read_csv(".//price5.csv")
    final_run = Test(p2)
    final_run.run_model(temp["Best n_list"], temp["Best n_clusters"], temp["Best n_effective"],
                        temp["Best step"], temp["Best threshold"], True)
