# -*- coding:utf-8 -*-
from random import sample
from math import exp
from diffprivlib.mechanisms import exponential, laplace


class Tree:
    def __init__(self):
        self.split_feature = None
        # 对于real value的条件为<，对于类别值得条件为=
        # 将满足条件的放入左树
        self.leftTree = None
        self.rightTree = None
        self.leafNode = None
        self.real_value_feature = True
        self.conditionValue = None

    def get_predict_value(self, instance):
        if self.leafNode:  # 到达叶子节点
            return self.leafNode.get_predict_value()
        if not self.split_feature:
            return self.leftTree.get_predict_value(instance)
        if self.real_value_feature and instance[self.split_feature] < self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)

    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftinfo = self.leftTree.describe()
        rightinfo = self.rightTree.describe()
        info = addtion_info+"{split_feature:"+str(self.split_feature) + \
               ",split_value:"+str(self.conditionValue) + \
               "[left_tree:"+leftinfo+",right_tree:"+rightinfo+"]}"
        return info

class LeafNode:
    def __init__(self):
        self.predictValue = None

    def get_predict_value(self):
        return self.predictValue

    def update_predict_value(self, idset, g, h, m, e, delta, p):
        sum_g = 0.0
        sum_h = 0.0
        for i in idset:
            sum_g += g[i]
            sum_h += h[i]
        if len(idset) > 0:
            self.predictValue = (-1 * sum_g/(sum_h + m))
        else:
            self.predictValue = 0.0

        # add Laplace noise
        if abs(self.predictValue) > p:
            lapnoise = laplace.Laplace(e, delta)
            self.predictValue = lapnoise.randomise(self.predictValue)


def construct_decision_tree(dataset, remainedSet, targets, e, delta, leaf_nodes, depth, max_depth, g, h, m, p, split_points=0):

    for it in range(2):
        gamma = 0.0001
        gainvalue = list()
        gainlist = dict()
        el = e / 2.0
        en = e / (2 * max_depth)

        if depth < max_depth:
            attributes = dataset.get_attributes()
            final_gain = -1
            final_remainder = -1
            selectedAttribute = None
            conditionValue = None
            selectedLeftIdSet = []
            selectedRightIdSet = []
            for attribute in attributes:
                is_real_type = dataset.is_real_type_field(attribute)
                attrValues = dataset.get_distinct_valueset(attribute)
                if is_real_type and split_points > 0 and len(attrValues) > split_points:
                    attrValues = sample(attrValues, split_points)
                for attrValue in attrValues:
                    leftIdSet = []
                    rightIdSet = []
                    for Id in remainedSet:
                        instance = dataset.get_instance(Id)
                        value = instance[attribute]
                        # 将满足条件的放入左子树
                        if (is_real_type and value < attrValue) or (not is_real_type and value == attrValue):
                            leftIdSet.append(Id)
                        else:
                            rightIdSet.append(Id)
                    g_left = []
                    g_right = []
                    h_left = []
                    h_right = []
                    for id in leftIdSet:
                        g_left.append(g[id])
                        h_left.append(h[id])
                    for id in rightIdSet:
                        g_right.append(g[id])
                        h_right.append(h[id])

                    gain = 1 / 2 * (sum(g_left) ** 2 / (sum(h_left) + m) +
                                    sum(g_right) ** 2 / (sum(h_right) + m) -
                                    (sum(g_left) + sum(g_right)) ** 2 / ((sum(h_left) + sum(h_right)) + m)) - gamma
                    gainvalue.append(gain)

                    if it == 0:
                        if gain > 0 and gain > final_gain:
                            selectedAttribute = attribute
                            conditionValue = attrValue
                            final_gain = gain
                            selectedLeftIdSet = leftIdSet
                            selectedRightIdSet = rightIdSet
                    elif it == 1:
                        selectedAttribute = attribute
                        value = dict()
                        value[0] = attribute
                        value[1] = attrValue
                        value[2] = leftIdSet
                        value[3] = rightIdSet
                        gainlist[gain] = value

            tree = Tree()
            privateset = dataset.get_privateattri()
            if not selectedAttribute:
                node = LeafNode()
                node.update_predict_value(remainedSet, g, h, m, el, delta, p)
                leaf_nodes.append(node)
                tree.leafNode = node
            elif it == 1:
                expnoise = exponential.Exponential(en, delta, gainvalue)
                id = expnoise.randomise()
                tree.split_feature = gainlist[gainvalue[id]][0]
                tree.real_value_feature = dataset.is_real_type_field(selectedAttribute)
                tree.conditionValue = gainlist[gainvalue[id]][1]
                tree.leftTree = construct_decision_tree(dataset, gainlist[gainvalue[id]][2], targets, e, delta,
                                                        leaf_nodes,
                                                        depth + 1, max_depth, g, h, m, p)
                tree.rightTree = construct_decision_tree(dataset, gainlist[gainvalue[id]][3], targets, e, delta,
                                                         leaf_nodes,
                                                         depth + 1, max_depth, g, h, m, p)
            elif selectedAttribute in privateset:
                # print("successfully add exp noise")
                continue
            else:
                tree.split_feature = selectedAttribute
                tree.real_value_feature = dataset.is_real_type_field(selectedAttribute)
                tree.conditionValue = conditionValue
                tree.leftTree = construct_decision_tree(dataset, selectedLeftIdSet, targets, e, delta, leaf_nodes,
                                                        depth + 1, max_depth, g, h, m, p)
                tree.rightTree = construct_decision_tree(dataset, selectedRightIdSet, targets, e, delta, leaf_nodes,
                                                         depth + 1, max_depth, g, h, m, p)
            return tree
        else:  # 是叶子节点
            node = LeafNode()
            node.update_predict_value(remainedSet, g, h, m, el, delta, p)
            leaf_nodes.append(node)
            tree = Tree()
            tree.leafNode = node
            return tree



