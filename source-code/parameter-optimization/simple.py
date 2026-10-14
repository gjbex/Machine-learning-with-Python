#!/usr/bin/env python

import optuna


def objective(trial):
    x = trial.suggest_float('x', -10.0, 10.0)
    y = trial.suggest_float('y', -10.0, 10.0)
    return (x - 2) ** 2 + (y + 3) ** 2


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print('Best trial:')
    trial = study.best_trial

    print(f'  Value: {trial.value}')
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
