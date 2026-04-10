# TFlaT

This README describes the training process of the transformer based flavortagger TFlaT.\
The provided scripts cover all steps that are required to get from the parquet files with training data to a weightfile with the trained model.
It also cover the computation of the effective tagging efficiency on a seperate test data set. 


---

## Setup the software

Create a conda environment by executing lines below:

```
git clone https://github.com/BenjaminSchwenker/tflat.git
cd tflat

conda create -n tflat  python=3.11.9
conda activate tflat

pip install 'tensorflow[and-cuda]'
# yaml, parquet, keras
```

---

## Training Samples

The Training of TFlaT requires $B^0 \rightarrow \nu \overline{\nu}$ samples to be produced. You can find the samples for training, validating and testing following this link.

---

## Hardware Requirements
The training process requires a CUDA capable GPU.\
The time needed to complete a training depends on the specific GPU. For a NVIDIA A100 GPU the expected time to completion with 10M training samples is ~1 days.\
Depending on the hardware used for the training some of the parameters found in the config.yaml file might need to be adjusted.\
If the VRAM of the used GPU is not sufficient, reduce the value of the 'batch_size' parameter.\
If the system memory is not sufficient, reduce the value of the 'chunk_size' parameter. Not that for optimal efficiency the 'chunk_size' should be an integer multiple of the 'batch_size'.\
If the GPU utilization is less than ~70% it might be possible to speed up the training by increasing the 'batch_size' parameter.\
The parameters set in the yaml config file are optimized for a A100 GPU and 120GB of system memory.

---

## Step-by-Step Guide

1. **Data preparation**

Download the files training_samples.parquet, validation_samples.parquet and test_samples.parquet from zenodo {add link here}.\
The training_samples.parquet file contains 10Mio training samples and the other two files contain 1Mio samples each.\ 

2. **Training**
    - To launch the training use the `trainer.py` script:
    ```bash
   python3 trainer.py --train_input /path/to/parquet/training_samples.parquet --val_input /path/to/parquet/validation_samples.parquet --configFile config.yaml
    ```
    - Should the training crash at any point it can be restarted from the latest checkpoint like this:
    ```bash
   python3 trainer.py --train_input /path/to/parquet/training_samples.parquet --val_input /path/to/parquet/validation_samples.parquet --config config.yaml --warmstart
    ```
   - Once the training is done a onnx weightfile is produced.

3. **Compute effective tagging efficiency**
   - The effective tagging efficiency is the main performance metric to evaluate flavor taggers at Belle II. 
   - See paper for details: https://doi.org/10.1140/epjc/s10052-022-10180-9
   - To compute the effective tagging efficiency on test data, run the next command
   ```bash
   python3 evaluate.py --test_input /path/to/parquet/test_samples.parquet --model model.keras --config config.yaml
   ```
