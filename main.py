# 1. PLA có thể cho vô số nghiệm khác nhau
# nếu hai class là linearly separable thì có vô số đường thằng phân cách 2 class đó
# Linear separable có nghĩa là một hyperplane (siêu phẳng), hyperplane này sẽ chia tập dữ liệu thành 2 phần sao cho tất
# cả các dữ liệu thuộc class thứ nhất sẽ nằm về một phía, và tất cả các dữ liệu thuộc class thứ hai sẽ nằm về một phía

# Trong trường hợp không hội tụ với dữ liệu gần linearly separable, hay nói cách khác là chi phí (cost) phải bỏ ra để
# tính toán sao cho dãy hội tụ là rất lớn, ta có thể cải tiến PLA theo thuật toán Pocket Algorithm, và giới hạn số lần
# lặp của PLA như sau

import numpy as np

np.random.seed(2)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis=1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
X = np.concatenate((np.ones((1, 2 * N)), X), axis=0)


def h(w, x):
    return np.sign(np.dot(w.T, x))


def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)  # True if h(w, X) == y else False


def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    pocket = []
    for idx in range(5):
        # mix data
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(3, 1)
            yi = y[0, mix_id[i]]

            # check if x[i] is miss_classified
            if h(w[-1], xi)[0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)

            if len(pocket) == 0:
                pocket.append([w_new, len(mis_points)])
            elif len(mis_points) < pocket[-1][-1]:
                    pocket.remove(pocket[-1])
                    pocket.append([w_new, len(mis_points)])

        # if has_converged(X, y, w[-1]):
        #     break
    return (w, pocket)


d = X.shape[0]
w_init = np.random.randn(d, 1)
w, mis_points = perceptron(X, y, w_init)
# print(mis_points)
print(w)
