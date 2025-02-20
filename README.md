# tflat_optuna


Tune hyperparameter of tflat using optuna library


Recipe for running on the GWDG compute cluster.

1) login to the frontend node

2) Create a conda environment by executing lines below:

```
git clone git@gitlab.desy.de:benjamin.schwenker/tflat_optuna.git
cd tflat_optuna

module load miniconda3
conda create -n tflat  python=3.11.9
conda activate tflat

pip install 'tensorflow[and-cuda]'
pip install optuna
```

3) Copy the data onto the GWDG cluster

The data file (data.npy) can be downloaded here to a local computer via browser

```
https://drive.google.com/file/d/12wZMJjv4BjLmYz97dgPgQz7cCKyxHeGq/view?usp=sharing
```

Next step is to upload it to the front end computer with rsync

```
scp -rp data.npy USERNAME@login-mdc.hpc.gwdg.de:/user/USER/USERNAME/tflat_optuna
```

4) Run the SLURM jobscript

```
sbatch jobscript.sh
```

You can repeat this command N times to have N parallel jobs on the cluster


5) Look at the results and find the best trial


```
python3 print_best.py
```

The history of the optimizations can be inferred from the backend file
optuna_journal_storage.log. You may uploat this file to web display at
https://optuna.github.io/optuna-dashboard/

