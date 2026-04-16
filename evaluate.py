#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from uncertainties import ufloat
import keras
import utils

rbins = [0.0, 0.1, 0.25, 0.45, 0.6, 0.725, 0.875, 1.0]
nbins = 7


def getwtf(data):
    ntot = len(data)
    nwro = len(data.query("tagflav != mctagflav"))
    w = nwro/ntot
    return ufloat(w, np.sqrt(w*(1-w))/np.sqrt(ntot))


def geteeff(data):
    ntot = len(data)
    splitdata = [data.query("r>"+str(rlo)+" & r<"+str(rup)) for rlo, rup in zip(rbins[:-1], rbins[1:])]
    eeff = 0
    for ri in range(nbins):
        w = getwtf(splitdata[ri])
        e = len(splitdata[ri])/ntot
        eeff += e*(1-2*w)**2
    return eeff


if __name__ == "__main__":

    # parse cli arguments
    parser = argparse.ArgumentParser(description='Evaluate TFlaT')
    parser.add_argument(  # input parser
        '--test_input',
        metavar='test_input',
        dest='test_input',
        type=str,
        default="dummyin_test.parquet",
        help='Path to testing parquet file'
    )
    parser.add_argument(  # input parser
        '--model',
        metavar='model',
        dest='model',
        type=str,
        default="model.keras",
        help='Path to trained model file'
    )
    parser.add_argument(  # input parser
        '--configFile',
        metavar='configFile',
        dest='configFile',
        type=str,
        default="config.yaml",
        help='Name of the config .yaml to be used and the produced weightfile'
    )

    args = parser.parse_args()
    testFile = args.test_input
    modelPath = args.model
    configFile = args.configFile

    model = keras.saving.load_model(modelPath)

    df = pd.read_parquet(testFile)

    config = utils.load_config(configFile)
    parameters = config['parameters']
    rank_variable = 'p'
    trk_variable_list = config['trk_variable_list']
    ecl_variable_list = config['ecl_variable_list']
    roe_variable_list = config['roe_variable_list']
    variables = utils.get_variables('pi+:tflat', rank_variable, trk_variable_list, particleNumber=parameters['num_trk'])
    variables += utils.get_variables('gamma:tflat', rank_variable, ecl_variable_list, particleNumber=parameters['num_ecl'])
    variables += utils.get_variables('pi+:tflat', rank_variable, roe_variable_list, particleNumber=parameters['num_roe'])
    target_variable = "qrCombined"

    X = df[variables].to_numpy()
    pred = model.predict(X)

    df['qrTFLAT'] = pred
    df['r'] = df['qrTFLAT'].abs()
    df['tagflav'] = np.ceil(df['qrTFLAT'])*2-1
    df['mctagflav'] = df[target_variable]

    print("wtf: ", getwtf(df))
    print("eeff: ", geteeff(df))
