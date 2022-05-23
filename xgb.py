# -*- coding:utf-8 -*-
from random import sample
from math import exp
from .tree import construct_decision_tree


class BinaryClassificationLoss():
    """分类的基类"""
    def compute_residual(self, dataset, subset, predicted):
        """计算残差"""
        residual = dict()
        for id in subset:
            y_i = dataset.get_instance(id)['diagnosis']
            residual[id] = 2.0 * y_i / (1 + exp(2 * y_i * predicted[id]))
        return residual

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F_{m-1}的值"""

    def initialize(self, f, subset):
        """初始化F_{0}的值"""
        for id in subset:
            f[id] = 0.0

    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值"""
        sum1 = sum([targets[id] for id in idset])
        sum2 = sum([abs(targets[id]) * (2 - abs(targets[id])) for id in idset])
        if sum2 == 0:
            return 0.0
        else:
            return sum1 / sum2

class MultiClassificationLoss():
    """分类的基类"""
    def compute_residual(self, subset, targets, predicted):
        """计算残差"""
        g = dict()
        h = dict()
        for id in subset:
            g[id] = - exp(-targets[id] * predicted[id]) * targets[id]
            h[id] = exp(-targets[id] * predicted[id]) * (targets[id] ** 2)
        return g, h

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F_{m-1}的值"""

    def initialize(self, f, subset, label_valueset):
        """初始化F_{0}的值"""
        for label in label_valueset:
            pred = dict()
            for id in subset:
                pred[id] = 0.0
            f[label] = pred

    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值"""


class RegressionLossFunction():
    def compute_residual(self, dataset, subset, predicted):
        """计算残差"""
        residual = dict()
        for id in subset:
            y_i = dataset.get_instance(id)['diagnosis']
            residual[id] = y_i - predicted[id]
        return residual

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F_{m-1}的值"""

    def initialize(self, f, subset):
        """初始化F_{0}的值"""
        for id in subset:
            f[id] = 0.0

    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值"""
        sum1 = sum([targets[id] for id in idset])
        if len(idset) == 0:
            return 0.0
        else:
            return sum1 / len(idset)


class GBDT:
    def __init__(self, sample_rate, learn_rate, max_depth, loss_type='binary-classification', split_points=0):
        self.sample_rate = sample_rate
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.loss_type = loss_type
        self.split_points = split_points
        self.loss = None

    def fit(self, dataset, train_data, trees, e, delta):
        if self.loss_type == 'multi-classification':
            pass
            # label_valueset = dataset.get_label_valueset()
            # self.loss = MultiClassificationLoss()
            #
            # subset = train_data
            # if 0 < self.sample_rate < 1:
            #     subset = sample(subset, int(len(subset) * self.sample_rate))
            #
            # trees = dict()
            # predicted = dict()
            # self.loss.initialize(predicted, subset, label_valueset)
            # if len(trees) >= 1:
            #     for it in range(1, len(trees)+1):
            #         for lab in label_valueset:
            #             for i in subset:
            #                 predicted[lab][i] += self.learn_rate * trees[it][lab].get_predict_value(dataset.get_instance(i))
            #
            # for label in label_valueset:
            #     targets = dict()
            #     for id in subset:
            #         l = dataset.get_instance(id)['diagnosis']
            #         if l == label:
            #             targets[id] = 1.0
            #         else:
            #             targets[id] = -1.0
            #     g, h = self.loss.compute_residual(subset, targets, predicted[label])
            #
            #     # 对某一个具体的label-K分类，选择max-depth个特征构造决策树
            #     leaf_nodes = []
            #     tree = construct_decision_tree(dataset, subset, predicted, leaf_nodes, 0, self.max_depth, g, h, self.split_points)
            #     trees[label] = tree
            # return trees
        else:
            if self.loss_type == 'binary-classification':
                self.loss = BinaryClassificationLoss()
            elif self.loss_type == 'regression':
                self.loss = RegressionLossFunction()

            # 这里取dataset的子集，后面要改为各自的数据集
            subset = train_data
            if 0 < self.sample_rate < 1:
                subset = sample(subset, int(len(subset)*self.sample_rate))

            # 用前面的树来预测subset的label值
            predicted = dict()  # 记录F_{m-1}的值
            self.loss.initialize(predicted, subset)
            if len(trees) >= 1:
                for it in range(1, len(trees)+1):
                    for i in subset:
                        predicted[i] += self.learn_rate * trees[it].get_predict_value(dataset.get_instance(i))

            # 用损失函数的负梯度作为回归问题提升树的残差近似值
            residual = self.loss.compute_residual(dataset, subset, predicted)

            leaf_nodes = []
            tree = construct_decision_tree(dataset, subset, residual, self.loss, e, delta, leaf_nodes, 0, self.max_depth, self.split_points)
            return tree

    def predict(self, dataset, test_data, trees):
        """
        对于回归和二元分类返回f值
        对于多元分类返回每一类的f值
        """
        predict = []
        if self.loss_type == 'binary-classification':
            for id in test_data:
                f_value = 0.0
                for iter in range(1, len(trees)+1):
                    if iter == len(trees):
                        f_value += trees[iter].get_predict_value(dataset.get_instance(id))
                    f_value += self.learn_rate * trees[iter].get_predict_value(dataset.get_instance(id))
                probs = dict()
                probs['1'] = 1 / (1 + exp(-2*f_value))
                probs['-1'] = 1 - probs['1']
                label = 1.0 if probs['1'] >= probs['-1'] else -1.0
                predict.append(label)
        elif self.loss_type =='multi-classification':
            for id in test_data:
                f_value = 0.0
                label_valueset = dataset.get_label_valueset()
                prob = dict()
                for label in label_valueset:
                    for iter in range(1, len(trees) + 1):
                        if iter == len(trees):
                            f_value += trees[iter][label].get_predict_value(dataset.get_instance(id))
                        f_value += self.learn_rate * trees[iter][label].get_predict_value(dataset.get_instance(id))
                    prob[label] = f_value
                label = min(label_valueset)
                target_value = prob[label]
                for lab in label_valueset:
                    if prob[lab] > target_value:
                        label = lab
                        target_value = prob[lab]
                predict.append(label)
        elif self.loss_type == 'regression':
            for id in test_data:
                f_value = 0.0
                for iter in range(1, len(trees)+1):
                    if iter == len(trees):
                        f_value += trees[iter].get_predict_value(dataset.get_instance(id))
                    f_value += self.learn_rate * trees[iter].get_predict_value(dataset.get_instance(id))
                predict.append(f_value)
        return predict


