#!/usr/bin/env python
# coding: utf8
""" Load dataset
    
    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
"""
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