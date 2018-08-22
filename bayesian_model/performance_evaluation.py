import matplotlib.pyplot as plt
import numpy as np
from empyrical import max_drawdown, alpha_beta
from matplotlib import rc
rc('mathtext', default='regular')


class Evaluation(object):

    def __init__(self, prices, n, dps, step, threshold,
                 benchmark_returns, holding_time, sample_size, long_term_invest):
        """
        prices: 价格数据
        n: 窗口长度, an integer
        dps: 平均价格变化
        step: 交易时间间隔
        thresohld: 进行交易的价格界限
        benchmark_returns: 市场平均回报率
        """
        self.prices = prices
        self.n = n
        self.dps = dps
        self.step = step
        self.threshold = threshold
        self.benchmark_returns = benchmark_returns
        self.holding_time = holding_time
        self.sample_size = sample_size
        self.long_term_invest = long_term_invest

    """
    建立虚拟账户并根据交易策略进行买卖
    返回最终账户余额和总投资金额
    """

    def visual_account(self, threshold):
        if type(threshold) == float:
            bank_balance = 2000
            position = 0
            investment = 0
            step_i = self.step
            profit = []
            price = []
            times = []
            final_profit = []
            hist_balance = []
            hist_cost = []
            for i in range(self.n, len(self.prices) - 1, step_i):
                current_price = self.prices[i]
                if self.long_term_invest:
                    if self.dps[i - self.n] > threshold:
                        position += 1
                        bank_balance -= self.prices[i]
                        final_profit.append(bank_balance - 2000 + position * current_price)
                        profit.append(bank_balance - 2000 + position * current_price)
                        price.append(self.prices[i])
                        times.append(i)
                        hist_cost.append(current_price - (final_profit[-1] - 2000))
                        investment += self.prices[i]
                        hist_balance.append(bank_balance)
                    if self.dps[i - self.n] < -threshold:
                        position -= 1
                        bank_balance += self.prices[i]
                        final_profit.append(bank_balance - 2000 + position * current_price)
                        profit.append(bank_balance - 2000 + position * current_price)
                        price.append(self.prices[i])
                        hist_cost.append(current_price - (final_profit[-1] - 2000))
                        hist_balance.append(bank_balance)
                        times.append(i)
                else:
                    if self.dps[i - self.n - 1] > threshold and position <= 0:
                        position += 1
                        bank_balance -= self.prices[i]
                        final_profit.append(bank_balance - 2000 + position * current_price)
                        profit.append(bank_balance - 2000 + position * current_price)
                        price.append(self.prices[i])
                        times.append(i)
                        hist_cost.append(current_price - (final_profit[-1] - 2000))
                        investment += self.prices[i]
                        hist_balance.append(bank_balance)
                    if self.dps[i - self.n - 1] < -threshold and position >= 0:
                        position -= 1
                        bank_balance += self.prices[i]
                        final_profit.append(bank_balance - 2000 + position * current_price)
                        profit.append(bank_balance - 2000 + position * current_price)
                        price.append(self.prices[i])
                        hist_cost.append(current_price - (final_profit[-1] - 2000))
                        times.append(i)
                        hist_balance.append(bank_balance)
            current_price1 = self.prices[len(self.prices) - 1]
            if position == 1:
                bank_balance += current_price1
                final_profit.append(bank_balance - 2000 + position * current_price1)
                times.append(len(self.prices))
                price.append(self.prices[-1])
                hist_cost.append(current_price1 - (final_profit[len(final_profit) - 1] - 2000))
                hist_balance.append(bank_balance)
            elif position == -1:
                bank_balance -= current_price1
                investment += current_price1
                final_profit.append(bank_balance - 2000 + position * current_price1)
                times.append(len(self.prices))
                price.append(self.prices[-1])
                hist_cost.append(current_price1 - (final_profit[len(final_profit) - 1] - 2000))
                hist_balance.append(bank_balance)
            else:
                bank_balance += position * current_price1
            return bank_balance, hist_balance, hist_cost, final_profit, price, times, profit, investment
        else:
            total_profit = []
            avg_profit = []
            holding_time = []
            sample_size = []
            for i in range(len(threshold[0])):
                bank_balance = 2000
                position = 0
                investment = 0
                times = []
                step_i = self.step
                final_profit = []
                profit = []
                trade_times = 0

                for j in range(self.n, len(self.prices) - 1, step_i):
                    current_price = self.prices[j]
                    if self.dps[j - self.n] > threshold[0, i] and position <= 0:
                        position += 1
                        bank_balance -= self.prices[j]
                        # profit.append(bank_balance - 2000 + position * current_price)
                        final_profit.append(bank_balance - 2000)
                        investment += self.prices[j]
                        trade_times += 1
                    if self.dps[j - self.n] < -threshold[0, i] and position >= 0:
                        position -= 1
                        bank_balance += self.prices[j]
                        # profit.append(bank_balance - 2000 + position * current_price)
                        final_profit.append(bank_balance - 2000)
                        trade_times += 1
                current_price1 = self.prices[len(self.prices) - 1]
                if position == 1:
                    bank_balance += current_price1
                    final_profit[-1] += current_price1
                    trade_times += 1
                if position == -1:
                    bank_balance -= current_price1
                    trade_times += 1
                    final_profit[-1] -= current_price1
                    investment += current_price1
                # final_profit = np.random.randn(1, len(profit))
                # final_profit[0] = profit[0]
                profit_sum = final_profit[-1]
                # for k in range(1, len(final_profit)):
                # final_profit[0, k] = final_profit[0, k - 1] + profit[k] - profit[k - 1]
                # profit_sum += profit[k]
                avg = profit_sum / len(final_profit)
                avg_profit.append(avg)
                avg_holding_time = (len(self.prices) - self.n) / trade_times
                total_profit.append(profit_sum)
                sample_size.append(trade_times)
                holding_time.append(avg_holding_time)
            return avg_profit, total_profit, sample_size, holding_time
    """
    计算最大回撤率
    返回最大回撤率，实际收益和按照Beta系数计算的期望收益之间的差额，业绩评价基准收益的总体波动性
    """

    def calculate_max_drawdown(self):
        temp = self.visual_account(self.threshold)
        final_balance = temp[0]
        cost = temp[7]
        balance = np.array(temp[1])
        initial_cost = np.array(temp[2])
        returns = balance / initial_cost
        alpha, beta = alpha_beta(returns, self.benchmark_returns)
        maxdrawdown = max_drawdown(returns)
        print("Balance: " + str(final_balance) + " Investment cost: " + str(cost))
        print('max drawdown = ' + str(maxdrawdown) + '; alpha = ' + str(alpha) + '; beta= ' + str(beta) + '.')
        return maxdrawdown, alpha, beta

    def plot_price_and_profit(self):
        temp = self.visual_account(self.threshold)
        bitcoin_price = temp[4]
        cum_profit = temp[3]
        time = temp[5]
        bitcoin_price = np.array(bitcoin_price)
        time = np.array(time)
        cum_profit = np.array(cum_profit)
        cum_profit = cum_profit.reshape(-1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time, bitcoin_price, '-', label='bitcoin price')
        ax2 = ax.twinx()
        ax2.plot(time, cum_profit, '-r', label='profit')
        ax.set_xlabel('time')
        ax.set_ylabel('bitcoin price(yuan)')
        ax2.set_ylabel('profit(yuan)')
        plt.legend(loc='upper right', ncol=2)
        plt.grid(True)
        plt.show()

    def plot_threshold_profit(self):
        """
        画出不同threshold和收益关系图
        """
        threshold = np.arange(0.1, 0.2, 0.001).reshape(1, -1)
        temp = self.visual_account(threshold)
        pnl = np.array(temp[0]).reshape(1, -1)
        total_pnl = np.array(temp[1]).reshape(1, -1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(threshold[0], pnl[0], '-b')
        ax.set_xlabel('threshold')
        ax.set_ylabel('pnl (blue)')
        ax2 = ax.twinx()
        ax2.plot(threshold[0], total_pnl[0], '-k')
        ax2.set_ylabel('total pnl (black)')
        plt.show()

    def plot_threshold_size(self):
        """
        画出不同threshold和平均收益关系图
        """
        threshold = np.arange(0.1, 0.2, 0.001).reshape(1, -1)
        temp = self.visual_account(threshold)
        holding_size = np.array(temp[3]).reshape(1, -1)
        sample_size = np.array(temp[2]).reshape(1, -1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(threshold[0], holding_size[0], '-b')
        ax.set_xlabel('threshold')
        ax.set_ylabel('pnl (blue)')
        ax2 = ax.twinx()
        ax2.plot(threshold[0], sample_size[0], '-k')
        ax2.set_ylabel('total pnl (black)')
        plt.show()

    def sharpe_ratio(self):
        """
        计算夏普比率
        """
        trade_price = self.visual_account(self.threshold)[4]
        point = self.visual_account(self.threshold)[6]
        c = (trade_price[-1] - trade_price[1]) / (point[-1] - point[1])
        sum_profit = np.sum(trade_price)
        mean = sum_profit / len(trade_price)
        b = 0
        for i in range(len(trade_price) - 1):
            b += (trade_price[i] - mean) ** 2
        sharpe_ratio = (sum_profit - c) / b
        return sharpe_ratio

    def fit(self, X, y=None):
        return self

    def get_params(self):
        return self.__dict__

    if __name__ == '__main__':
        import pandas as pd

        p1 = pd.read_csv('.//p1.csv')
        p2 = pd.read_csv('.//p2.csv')
        p3 = pd.read_csv('.//p3.csv')
