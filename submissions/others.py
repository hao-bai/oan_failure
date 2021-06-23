#!/usr/bin/env python
# coding: utf8
''' Load dataset
    
    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
'''
import os
import pathlib
import rampwf as rw
from contextlib import contextmanager


@contextmanager
def cd(newdir):
    if isinstance(newdir, pathlib.Path):
        newdir = str(newdir)
    prevdir = os.getcwd()  # save current working path
    os.chdir(os.path.expanduser(newdir))  # change directory
    try:
        yield
    finally:
        os.chdir(prevdir)  # revert to the origin workinng path

def load_all_dataset():
    with cd("~/Codes/HuaweiRAMP"):
        problem = rw.utils.assert_read_problem()
        X_train, y_train = problem.get_train_data()
        X_test, y_test = problem.get_test_data()
    return X_train, y_train, X_test, y_test

def rename_dataset(fe, X_train, y_train, X_test, y_test, show_imbalance=False):
    # 重命名
    from copy import deepcopy
    # 训练集
    print("==== TRAIN SET ====")
    X_source = deepcopy( fe.transform(X_train.source) )
    print("  | X_source:", X_source.shape, end=" ; ")
    y_source = deepcopy( y_train.source )
    print("y_source:", y_source.shape)
    if show_imbalance:
        tmp1 = y_source[y_source==1].shape[0]
        tmp0 = y_source[y_source==0].shape[0]
        total = y_source.shape[0]
        print("  | \timbalance: {}({:.1f}%) failure, {}({:.1f}%) weak".format(
            tmp1, tmp1/total*100, tmp0, tmp0/total*100, ))
    X_source_bkg = deepcopy( fe.transform(X_train.source_bkg) )
    print("A | X_source_bkg:", X_source_bkg.shape)
    X_target = deepcopy( fe.transform(X_train.target) )
    print("----")
    print("  | X_target:", X_target.shape, end=" ; ")
    y_target = deepcopy( y_train.target )
    print("y_target:", y_target.shape)
    if show_imbalance:
        tmp1 = y_target[y_target==1].shape[0]
        tmp0 = y_target[y_target==0].shape[0]
        total = y_target.shape[0]
        print("  | \timbalance: {}({:.1f}%) failure, {}({:.1f}%) weak".format(
            tmp1, tmp1/total*100, tmp0, tmp0/total*100, ))
    X_target_bkg= deepcopy( fe.transform(X_train.target_bkg) )
    print("B | X_target_bkg:", X_target_bkg.shape)
    X_target_unlabeled = deepcopy( fe.transform(X_train.target_unlabeled) )
    print("  | X_target_unlabeled:", X_target_unlabeled.shape)
    # 测试集
    print("==== TEST SET ====")
    X_test.target = fe.transform(X_test.target)
    print("  | X_test.target:", X_test.target.shape, end=" ; ")
    print("y_test.target:", y_test.target.shape)
    if show_imbalance:
        tmp1 = y_test.target[y_test.target==1].shape[0]
        tmp0 = y_test.target[y_test.target==0].shape[0]
        total = y_test.target.shape[0]
        print("  | \timbalance: {}({:.1f}%) failure, {}({:.1f}%) weak".format(
            tmp1, tmp1/total*100, tmp0, tmp0/total*100, ))
    X_test.target_bkg = fe.transform(X_test.target_bkg)
    print("B | X_test.target_bkg:", X_test.target_bkg.shape)
    print("  | X_test.target_unlabeled:", X_test.target_unlabeled)
    return [X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target, X_test]