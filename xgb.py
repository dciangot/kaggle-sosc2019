#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : xgb.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: 
"""

# system modules
import os
import sys
import time
import json
import pprint
import pickle

# pandas modules
import pandas as pd

# numpy modules
import numpy as np

# xgboost modules
import xgboost as xgb

# sklearn modules
from sklearn import metrics

# matplotlib
import matplotlib.pyplot as plt

# local modules
from utils import preprocess, OptionParser, plot_roc_curve

def lookup(x, w):
    "Helper function to look-up embeddings values from weights matrix"
    return w[x]

def merge_embeddings(xdf, cat, emb, w, verbose):
    """
    Helper function to merge embeddings values into given dataframe
    """
    if verbose:
        print("embeddings category: '%s' embeddings shape %s" % (cat, np.shape(w)))
    if cat not in list(xdf.columns):
        print("categorical variable '%s' not found in data-frame" % cat)
        print(type(xdf), np.shape(xdf), list(xdf.columns))
        sys.exit(1)

    # perform look-up in our weights matrix for given value of categorical variables
    emd_values = list(xdf[cat].apply(lookup, args=(w,)))

    # create vector of names for embeddings columns
    emd_names = ['%s_%s' % (cat, i) for i in range(len(w[0]))]

    if verbose:
        print("new embeddings columns", len(emd_names), "with shape", np.shape(emd_values))
    emd_df = pd.DataFrame(emd_values, columns=emd_names)
    if verbose:
        print("xdf dim", np.shape(xdf), "emb dim", np.shape(emd_df))

    # merge emdedded df with our data-frame column-wise, to stick two dataframes
    # together we reset their indexes
    names = list(xdf.columns) + emd_names
    xdf.reset_index(drop=True, inplace=True)
    emd_df.reset_index(drop=True, inplace=True)
    xdf = pd.concat([xdf, emd_df], axis=1, ignore_index=True)
    xdf.columns = names

    # drop categorical variable from our data-frame since we replace it with
    # emdedded values
    if cat in list(xdf.columns):
        xdf = xdf.drop(cat, axis=1)
    if verbose:
        print("new dimension of dataframe:", np.shape(xdf), type(xdf))
    return xdf

def add_weights(X_train, X_valid, X_test, weights, emb, verbose=False):
    "Add embedding weights to our dataframes"
    if not weights:
        return X_train, X_valid, X_test
    for name in os.listdir(weights):
        if name.endswith('pkl'):
            fname = '%s/%s' % (weights, name)
            with open(fname, 'rb') as istream:
                # load weight matrix, it is in form of [array([[], [])]
                w = pickle.load(istream)[0]
                if verbose:
                    print("loading", fname)
                # name of categorical variable
                cat = name.split('.')[0]
                X_train = merge_embeddings(X_train, cat, emb, w, verbose)
                X_valid = merge_embeddings(X_valid, cat, emb, w, verbose)
                X_test = merge_embeddings(X_test, cat, emb, w, verbose)
    return X_train, X_valid, X_test

def model(fout, X_train, X_valid, X_test, y_train, y_valid, ids, cat_vars, cat_sz, emb_szs, params, verbose):
    "Build and fit model for given train/validation/test/out files"

    nround = params.get('nround', 10)
    early_stopping_rounds = params.get('early_stopping_rounds', 50)


    drop_this = [
            "PersonalField41",
            "PersonalField37",
            "PropertyField10",
            "PersonalField46",
            "GeographicField23A",
            "PersonalField32",
            "GeographicField21A",
            "GeographicField64_2",
            "GeographicField56A",
            "PersonalField51",
            "PersonalField52",
            "PersonalField30",
            "PersonalField71",
            "PersonalField68",
            "GeographicField22A",
            "PersonalField7_1",
            "PersonalField47",
            "Field12_1",
            "PropertyField2A",
            "PersonalField62",
            "PropertyField11A",
            "GeographicField64_0",
            "GeographicField63_0",
            "GeographicField5A",
            "PersonalField29",
            "SalesField9",
            "PersonalField72",
            "PersonalField23",
            "GeographicField60A",
            "PersonalField44",
            "GeographicField12A",
            "PersonalField78",
            "PersonalField48",
            "PersonalField58",
            "GeographicField13A",
            "PropertyField4_1",
            "PropertyField4_0",
            "PersonalField33",
            "GeographicField62A",
            "PropertyField36_1",
            "PersonalField74",
            "PropertyField38_0",
            "PersonalField36",
            "PersonalField50",
            "GeographicField61A",
            "PersonalField54",
            "PersonalField53",
            "PropertyField30_1",
            "PropertyField22",
            "PersonalField38",
    "PersonalField55",
    "GeographicField63_1",
    "GeographicField18A",
    "PropertyField38_1",
    "GeographicField64_1",
    "PropertyField30_0",
    "Field12_0",
    "SalesField13",
    "PersonalField59",
    "PersonalField56",
    "PropertyField28_2",
    "SalesField15",
    "PersonalField19_23",
    "PersonalField76",
    "PropertyField31_2",
    "SalesField14",
    "Field6_3",
    "PropertyField36_0",
    "PersonalField19_21",
    "PersonalField57",
    "GeographicField15A",
    "PersonalField63",
    "PropertyField23",
    "PersonalField7_0",
    "Field6_4",
    "Field6_2",
    "PropertyField14_1",
    "PersonalField75",
    "PropertyField13",
    "PropertyField11B",
    "Field6_1",
    "PersonalField19_22",
    "PersonalField19_24",
    "PersonalField31",
    "PersonalField19_8",
    "PersonalField19_20",
    "PropertyField28_0",
    "PersonalField77",
    "PersonalField61",
    "PersonalField25",
    "PersonalField17_4",
    "PersonalField19_17",
    "PropertyField32_1",
    "GeographicField7A",
    "PersonalField19_5",
    "year_1",
    "PropertyField3_1",
    "PersonalField19_12",
    "PropertyField14_2",
    "PersonalField19_10",
    "GeographicField12B",
    "GeographicField11A",
    "PersonalField18_21",
    "PersonalField79",
    "PropertyField17",
    "PropertyField28_1",
    "PersonalField19_18",
    "PersonalField19_3",
    "PersonalField80",
    "PropertyField7_5",
    "PersonalField19_4",
    "PropertyField15",
    "PropertyField7_2",
    "PersonalField19_7",
    "PersonalField18_12",
    "PersonalField19_0",
    "PersonalField18_20",
    "PersonalField19_25",
    "Field6_0",

            ]
    
    for col in drop_this:
        X_train = X_train.drop(col, axis=1)
        X_valid = X_valid.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)

    TOPF = ['PersonalField10A', 'SalesField1A', 'PersonalField9', 'SalesField1B', 'PersonalField10B']


    X_train["avg"] = X_train.mean(axis=1)
    X_valid["avg"] = X_valid.mean(axis=1)
    X_test["avg"]  = X_test.mean(axis=1)

    X_train["sumTop"] = X_train[TOPF].sum(axis=1)
    X_valid["sumTop"] = X_valid[TOPF].sum(axis=1)
    X_test["sumTop"]  = X_test[TOPF].sum(axis=1)

    ncols = X_test.columns.size

    X_train['value_count'] = X_train.apply(lambda x: (ncols - x.count())/ncols, axis=1)
    X_valid['value_count'] = X_valid.apply(lambda x: (ncols - x.count())/ncols, axis=1)
    X_test['value_count'] = X_test.apply(lambda x: (ncols - x.count())/ncols, axis=1)
    
    X_train["comb1"] = X_train["sumTop"]*X_train["value_count"]
    X_valid["comb1"] = X_valid["sumTop"]*X_valid["value_count"]
    X_test["comb1"]  = X_test["sumTop"]*X_test["value_count"]

    X_train["comb2"] = X_train["sumTop"]*X_train["avg"]
    X_valid["comb2"] = X_valid["sumTop"]*X_valid["avg"]
    X_test["comb2"]  = X_test["sumTop"]*X_test["avg"]

    X_train["comb5"] = X_train["sumTop"]-X_train["value_count"]
    X_valid["comb5"] = X_valid["sumTop"]-X_valid["value_count"]
    X_test["comb5"]  = X_test["sumTop"]-X_test["value_count"]

    X_train["comb6"] = X_train["sumTop"]-X_train["avg"]
    X_valid["comb6"] = X_valid["sumTop"]-X_valid["avg"]
    X_test["comb6"]  = X_test["sumTop"]-X_test["avg"]

    newTOP = [
            'SalesField8',   'SalesField6',  'PersonalField10A',   'SalesField2B', 'PersonalField10B',
            'PropertyField29',   'SalesField5',  'sumTop', 'SalesField1B',   'PersonalField9'
            ]

    X_train["avgTop2"] = X_train[newTOP].mean(axis=1)
    X_valid["avgTop2"] = X_valid[newTOP].mean(axis=1)
    X_test["avgTop2"]  = X_test[newTOP].mean(axis=1)

    X_train["sumTop2"] = X_train[newTOP].sum(axis=1)
    X_valid["sumTop2"] = X_valid[newTOP].sum(axis=1)
    X_test["sumTop2"]  = X_test[newTOP].sum(axis=1)

    print("Train shape", np.shape(X_train))
    print("Valid shape", np.shape(X_valid))
    print("Test  shape", np.shape(X_test))

    # preparre DMatrix object for training/fitting
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)

    # model parameters
    args = {'max_depth': 6,
            'eta':0.012,
            'subsample':0.86,
            'colsample_bytree': 0.38,
            'eval_metric': 'auc',
            'silent':0,
            'n_jobs':4,
            'objective':'binary:logistic'}
    if verbose:
        print("model parameters")
        pprint.pprint(args)

    # use evaluation list while traning
    evallist  = [(deval,'eval'), (dtrain,'train')]
    # train our model with early stopping that we'll see that we don't overfit
    #bst = xgb.train(args, dtrain, nround, evallist, early_stopping_rounds=early_stopping_rounds)
    bst = xgb.train(args, dtrain, nround, evallist, early_stopping_rounds=early_stopping_rounds)

    # try eli5 explanation of our model
    # see permutation importance:
    # https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights
    try:
        import eli5
        html_obj = eli5.show_weights(bst, top=10)
        import html2text
        print(html2text.html2text(html_obj.data))
    except:
        pass

    # validate results
    pred = bst.predict(deval)
    myscores = bst.get_score()
    for i in sorted(myscores, key=myscores.get):
        print("\""+i+"\","+str(myscores[i]))
    print("AUC", metrics.roc_auc_score(y_valid, pred))

    # create AUC/ROC plot
    plot_roc_curve(y_valid, pred)


    # make prediction
    if fout:
        #pred = sclf2.predict(X_test)
        #pred = sclf.predict(X_test)
        #pred = eclf1.predict(X_test)
        pred = bst.predict(dtest)
        data = {'QuoteNumber':ids, 'QuoteConversion_Flag': pred}
        sub = pd.DataFrame(data, columns=['QuoteNumber', 'QuoteConversion_Flag'])
        print("Write prediction to %s" % fout)
        sub.to_csv(fout, index=False)

def run(ftrain, fvalid, ftest, fout, date_col, params, weights, verbose):
    "Main function we run over our dataset"
    print("Train file     : %s" % ftrain)
    print("Validation file: %s" % fvalid)
    print("Test file      : %s" % ftest)
    print("Date column    : %s" % date_col)
    print("Parameters     : %s" % json.dumps(params))

    # preprocess our data
    X_train, X_valid, X_test, y_train, y_valid, ids, cat_vars, cat_sz, emb_szs = \
            preprocess(ftrain, fvalid, ftest, date_col, params, verbose)

    if verbose:
        print("Categorical cardinality")
        print(cat_sz)
        print("Embedding matrix")
        print(emb_szs)

    # if we provides embedding weights we'll add them to our data frames
    if weights:
        X_train, X_valid, X_test = add_weights(X_train, X_valid, X_test, weights, cat_sz, verbose)

    # construct and fit the model
    model(fout, X_train, X_valid, X_test, y_train, y_valid, ids, cat_vars, cat_sz, emb_szs, params, verbose)

def main():
    "Main function"
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    time0 = time.time()
    params = json.load(open(opts.params)) if opts.params else {}
    run(opts.ftrain, opts.fvalid, opts.ftest, opts.fout, opts.date_col, params, opts.weights, opts.verbose)
    print("Done in %s sec" % (time.time()-time0))

if __name__ == '__main__':
    main()

