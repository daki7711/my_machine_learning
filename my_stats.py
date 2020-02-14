import numpy as np
from numpy import mean
from numpy import std


class TwoArrayStats:
    def __init__(self, array1, array2):
        self.x = array1
        self.y = array2

    def calculate_rank(self, vector):
        a = {}
        rank = 1
        for num in sorted(vector):
            if num not in a:
                a[num] = rank
                rank = rank + 1
        return [a[i] for i in vector]

    #cov(X, Y) = (sum (x - mean(X)) * (y - mean(Y)) ) * 1/(n-1)
    def covariance_of_two_array(self, x, y):
        x_mean = mean(x)
        y_mean = mean(y)

        sum = 0
        for i in range(len(y)):
            sum += (x[i] - x_mean) * (y[i] - y_mean)

        return (1 / len(y)) * sum

    # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
    def Pearson(self):
        return self.covariance_of_two_array(self.x,self.y) / (std(self.x) * std(self.y))

    # Spearman's correlation coefficient = covariance(rank(X), rank(Y)) / (stdv(rank(X)) * stdv(rank(Y)))
    def Spearman(self):
        x = np.asarray(self.calculate_rank(self.x))
        y = np.asarray(self.calculate_rank(self.y))

        return (self.covariance_of_two_array(x, y)) / (std(x) * std(y))

    def Rsquare(self):
        return self.Pearson() * self.Pearson()


def main():
    test = TwoArrayStats([1,2,3], [1,2,3])
    print(test.Pearson())
    print(test.Spearman())


if __name__ == '__main__':
    main()
