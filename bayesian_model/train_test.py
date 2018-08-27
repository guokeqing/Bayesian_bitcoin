import numpy as np
import pandas as pd
from bayesian_model.bayesian_regression import Prediction
from bayesian_model.data_processor import ProcessData


class Train(object):
    def __init__(self, s, prices, n, price_3):
        self.prices = prices
        self.s = s
        self.n = n
        self.price_3 = price_3

    def intensive_training(self):
        total_w0 = 0
        total_w1 = 0
        total_w2 = 0
        total_w3 = 0
        total_w4 = 0
        for i in range(0, 10):
            index = round(len(self.prices) / 10)
            train_i = self.prices[i * index:(i + 1) * index]
            pre = Prediction(self.s, train_i, self.n, self.price_3)
            [w0, w1, w2, w3, w4] = pre.find_parameters_w()
            total_w0 += w0
            total_w1 += w1
            total_w2 += w2
            total_w3 += w3
            total_w4 += w4
        best_w0 = total_w0 / 10
        best_w1 = total_w1 / 10
        best_w2 = total_w2 / 10
        best_w3 = total_w3 / 10
        best_w4 = total_w4 / 10
        return best_w0, best_w1, best_w2, best_w3, best_w4

    def error(self):
        pre = Prediction(self.s, self.prices, self.n, self.price_3)
        predicted_dp = pre.predict_delta_p()
        print(predicted_dp)
        """
        actual_dp = [0]
        variance = 0
        for i in range(max(self.n), len(self.price_3)-1):
            actual_dp.append(self.price_3[i] - self.price_3[i - 1])
            variance += (predicted_dp[i - max(self.n)] - actual_dp[i - max(self.n)]) ** 2
        error = variance / (2 * len(predicted_dp))
        
        return error
        """

if __name__ == '__main__':
    p1 = pd.read_csv(".//price4.csv")
    p2 = pd.read_csv(".//price5.csv")
    p3 = pd.read_csv(".//price6.csv")
    price_reshaped_1 = p1.values.reshape((1, -1))[0, :]
    price_reshaped_2 = p2.values.reshape((1, -1))[0, :]
    price_reshaped_3 = p3.values.reshape((1, -1))[0, :]
    n = [90, 180, 360, 720]
    process = ProcessData(price_reshaped_1, n, 100, 20)
    s = process.select_effective_clusters()
    temp = Train(s, price_reshaped_2, n, price_reshaped_3)
    result = temp.error()
    print(result)