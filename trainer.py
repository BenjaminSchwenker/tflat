#!/usr/bin/env python3


import os
import argparse


if __name__ == "__main__":

    import keras
    from fitter import fit
    import utils
    from model import get_tflat_model

    # parse cli arguments
    parser = argparse.ArgumentParser(description='Train TFlaT')
    parser.add_argument(  # input parser
        '--train_input',
        metavar='train_input',
        dest='train_input',
        type=str,
        default="dummyin_train.parquet",
        help='Path to training parquet file'
    )
    parser.add_argument(  # input parser
        '--val_input',
        metavar='val_input',
        dest='val_input',
        type=str,
        default="dummyin_val.parquet",
        help='Path to validation parquet file'
    )
    parser.add_argument(  # input parser
        '--configFile',
        metavar='configFile',
        dest='configFile',
        type=str,
        default="config.yaml",
        help='Name of the config .yaml to be used and the produced weightfile'
    )
    parser.add_argument(  # checkpoint parser
        '--checkpoint',
        metavar='checkpoint',
        dest='checkpoint',
        type=str,
        nargs='+',
        default="./ckpt/checkpoint.model.keras",
        help='Path to checkpoints'
    )
    parser.add_argument(
        '--warmstart',
        help='Start from checkpoint',
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()

    train_file = args.train_input
    val_file = args.val_input
    checkpoint_filepath = args.checkpoint
    warmstart = args.warmstart
    configFile = args.configFile

    config = utils.load_config(configFile)
    parameters = config['parameters']
    rank_variable = 'p'
    trk_variable_list = config['trk_variable_list']
    ecl_variable_list = config['ecl_variable_list']
    roe_variable_list = config['roe_variable_list']
    variables = utils.get_variables('pi+:tflat', rank_variable, trk_variable_list, particleNumber=parameters['num_trk'])
    variables += utils.get_variables('gamma:tflat', rank_variable, ecl_variable_list, particleNumber=parameters['num_ecl'])
    variables += utils.get_variables('pi+:tflat', rank_variable, roe_variable_list, particleNumber=parameters['num_roe'])

    if not warmstart:
        if os.path.isfile(checkpoint_filepath):
            os.remove(checkpoint_filepath)

        model = get_tflat_model(parameters=parameters, number_of_features=len(variables))

        # configure the optimizer
        cosine_decay_scheduler = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config['initial_learning_rate'],
            decay_steps=config['decay_steps'],
            alpha=config['alpha']
        )

        optimizer = keras.optimizers.AdamW(
            learning_rate=cosine_decay_scheduler, weight_decay=config['weight_decay']
        )

        # compile the model
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.binary_crossentropy,
            metrics=[
                'accuracy',
                keras.metrics.AUC(),
                keras.metrics.MeanSquaredError()])
    else:
        model = keras.models.load_model(checkpoint_filepath)

    model.summary()

    fit(
        model,
        train_file,
        val_file,
        "tflat_variables",
        variables,
        "qrCombined",
        config,
        checkpoint_filepath
    )

    model.save('model.keras')
