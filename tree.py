# -*- coding:utf-8 -*-
from random import sample
from math import exp
from diffprivlib.mechanisms import exponential, laplace

def MSE(values):
    """
    均平方误差 mean square error
    """
    if len(values) < 2:
        return 0
    mean = sum(values)/float(len(values))
    error = 0.0
    for v in values:
        error += (mean-v)*(mean-v)
    return error

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
            raise ValueError("the tree is null")
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

    def update_predict_value(self, targets, idset, loss, e, delta):
        self.predictValue = loss.update_ternimal_regions(targets, idset)
        # add Laplace noise
        lapnoise = laplace.Laplace(e, delta)
        self.predictValue = lapnoise.randomise(self.predictValue)

def construct_decision_tree(dataset, remainedSet, targets, loss, e, delta, leaf_nodes, depth, max_depth, split_points=0):
    gainvalue = list()
    gainlist = dict()
    el = e / 2.0
    en = e / (2 * max_depth)

    if depth < max_depth:
        attributes = dataset.get_attributes()
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
                    if (is_real_type and value < attrValue)or(not is_real_type and value == attrValue):
                        leftIdSet.append(Id)
                    else:
                        rightIdSet.append(Id)
                leftTargets = [targets[id] for id in leftIdSet]
                rightTargets = [targets[id] for id in rightIdSet]
                gain = MSE(leftTargets) + MSE(rightTargets)
                gainvalue.append(gain)
                value = dict()
                value[0] = attribute
                value[1] = attrValue
                value[2] = leftIdSet
                value[3] = rightIdSet
                gainlist[gain] = value

        expnoise = exponential.Exponential(en, delta, gainvalue)
        id = expnoise.randomise()
        tree = Tree()
        tree.split_feature = gainlist[gainvalue[id]][0]
        tree.real_value_feature = dataset.is_real_type_field(gainlist[gainvalue[id]][0])
        tree.conditionValue = gainlist[gainvalue[id]][1]
        tree.leftTree = construct_decision_tree(dataset, gainlist[gainvalue[id]][2], targets, loss, e, delta, leaf_nodes, depth + 1,
                                                max_depth)
        tree.rightTree = construct_decision_tree(dataset, gainlist[gainvalue[id]][3], targets, loss, e, delta, leaf_nodes, depth + 1,
                                                max_depth)
        return tree
    else:  # 是叶子节点
        node = LeafNode()
        node.update_predict_value(targets, remainedSet, loss, el, delta)
        leaf_nodes.append(node)
        tree = Tree()
        tree.leafNode = node
        return tree
