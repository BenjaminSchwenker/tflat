import optuna
import numpy as np
import tensorflow as tf

from tensorflow_tflat_model import get_tflat_model


def objective(trial):

    # Clear clutter from previous Keras session graphs.
    tf.keras.backend.clear_session()

    with open('data.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)
        Xtest = np.load(f)
        ytest = np.load(f)

    # get number of features
    number_of_features = X.shape[1]

    # set random state
    tf.random.set_seed(1234)

    parameters = {}
    parameters['num_tracks'] = 10
    parameters['num_features'] = 14
    parameters["num_transformer_blocks"] = 3
    parameters["num_heads"] = 4
    parameters["embedding_dims"] = 128
    parameters["mlp_hidden_units_factors"] = [2, 1,]
    parameters["dropout_rate"] = trial.suggest_float("dropout_rate", 0, 0.8) # 0.2

    model = get_tflat_model(parameters, number_of_features)

    batch_size = 256
    num_steps = int(X.shape[0]/batch_size)
    epochs = 100
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True)
    initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-1, log=True)
    decay_steps = trial.suggest_int("decay_steps", 1, 100*num_steps, log=True)
    alpha = trial.suggest_float("alpha", 1e-7, 1e-1, log=True)

    cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=cosine_decay_scheduler, weight_decay=weight_decay
    )


    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC()])

    model.summary()

    
    # perform fit() with early stopping callback
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=7,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True)]

    model.fit(X, y, validation_data=(Xtest, ytest), batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=2)

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(Xtest, ytest, verbose=0)
    val_accuracy = score[1]

    return val_accuracy


if __name__ == "__main__":

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend("./optuna_journal_storage.log"),
    )

    study = optuna.create_study(study_name="tflat-study", direction="maximize", storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=5, gc_after_trial=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
