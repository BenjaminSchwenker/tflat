from path_explain import PathExplainerTF
import explainer_utils
from model import *
import argparse
import utils
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import keras


if __name__ == "__main__":

    # parse cli arguments
    parser = argparse.ArgumentParser(description='Evaluate TFlaT')
    parser.add_argument(  # input parser
        '--data',
        metavar='data',
        dest='data',
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
    parser.add_argument(  # input parser
        '--output',
        metavar='output',
        dest='output',
        type=str,
        default='attributions.npz',
        help='Name of the output npz file containing the attributions and the data used',
    )

    # Read the arguments from the parser
    args = parser.parse_args()
    dataFile = args.data
    modelPath = args.model
    configFile = args.configFile
    outputFile = args.output

    # Get the values from the config file
    config = explainer_utils.load_config(configFile)
    batch_size = config['batch_size']
    num_steps = config['num_steps']
    use_expectation = config['use_expectation']
    num_samples = config['num_samples']
    parameters = config['parameters']
    rank_variable = 'p'
    trk_variable_list = config['trk_variable_list']
    ecl_variable_list = config['ecl_variable_list']
    roe_variable_list = config['roe_variable_list']
    variables = utils.get_variables('pi+:tflat', rank_variable, trk_variable_list, particleNumber=parameters['num_trk'])
    variables += utils.get_variables('gamma:tflat', rank_variable, ecl_variable_list, particleNumber=parameters['num_ecl'])
    variables += utils.get_variables('pi+:tflat', rank_variable, roe_variable_list, particleNumber=parameters['num_roe'])

    # Load the parquet file
    df = pq.ParquetFile(dataFile)

    # Take samples from the parquet file
    samples = []
    collected = 0
    for batch in df.iter_batches(batch_size=num_samples, columns=variables):
        df = batch.to_pandas()

        remaining = num_samples - collected
        if remaining <= 0:
            break

        take = min(len(df), remaining)
        samples.append(df.sample(n=take))

        collected += take
    data = pd.concat(samples).iloc[:num_samples].to_numpy().astype(np.float32)

    print(f'\n\n Created data sample with shape: {data.shape}')

    # Load the model
    model = keras.saving.load_model(modelPath)
    print("Model loaded successfully.")

    # Create the explainer
    explainer = PathExplainerTF(model)
    print("Explainer created successfully.")

    # Create the baseline with the mean of the data
    baseline = np.mean(data, axis=0, keepdims=True)

    # Calculate the attributions
    print('Calculating attributions...')
    attributions = explainer.attributions(inputs=data,
                                          baseline=baseline,
                                          batch_size=batch_size,
                                          num_samples=num_steps,
                                          use_expectation=use_expectation,
                                          verbose=True)

    # Save the attributions into an npz file
    np.savez_compressed(
        outputFile,
        attributions=attributions[0],
        data=data,
    )
    print(f"Saved attributions to {outputFile} with shape: {attributions[0].shape}")
