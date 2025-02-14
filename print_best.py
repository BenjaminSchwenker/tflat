import optuna


if __name__ == "__main__":

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend("./optuna_journal_storage.log"),
    )

    study = optuna.create_study(study_name="tflat-study", direction="maximize", storage=storage, load_if_exists=True)
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
