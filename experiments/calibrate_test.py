import numpy
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.hydrodata import load_and_process_data, store_results
from conceptual_models.lumped import *
from hydroutils import Calibration, ObjectiveFunction


def get_params_class(model_name):
    if model_name == 'exp-hydro':
        return ExphydroParameters()

    if model_name == 'hbv':
        return HBVParameters()

    if model_name == 'abcd':
        return ABCDParameters()

    if model_name == 'flex':
        return FLEXParameters()

    if model_name == 'wbm':
        return WBMParameters()


def get_model(model_name, p, pet, t):
    if model_name == 'exp-hydro':
        return ExphydroModel(p, pet, t)

    if model_name == 'hbv':
        return HBVModel(p, pet, t)

    if model_name == 'abcd':
        return ABCDModel(p, pet, t)

    if model_name == 'flex':
        return FLEXAModel(p, pet, t)

    if model_name == 'wbm':
        return WBModel(p, pet, t)


def calibrate_test(working_path, data_root_path, model_name, basin_id, dataset_bounds, n_part=10,
                   warmup=365):
    # print
    print(f'Basin ID: {basin_id}\n')
    # create path
    save_root = f'{working_path}/results/{model_name}/{basin_id}'
    Path(save_root).mkdir(parents=True, exist_ok=True)
    # load and process data
    train_x, train_y, test_x, test_y, test_set = load_and_process_data(data_root_path, basin_id, dataset_bounds)
    ###  CALIBRATION  ###
    # 1. data
    p_train, pet_train, t_train, qobs_train = train_x[:, 0], train_x[:, 1], train_x[:, 2], train_y[:, 0]

    # 2. parameters
    params = [get_params_class(model_name) for j in range(n_part)]

    # 3. initialise the model by loading its climate inputs
    model = get_model(model_name, p_train, pet_train, t_train)

    # 4. calibrate the model
    calperiods_obs = [warmup, len(train_y) - 1]
    calperiods_sim = [warmup, len(train_y) - 1]

    paramsmax = Calibration.pso_maximise(model, params, qobs_train,
                                         ObjectiveFunction.nashsutcliffe,
                                         calperiods_obs, calperiods_sim)
    print('Calibration run NSE value = ', paramsmax.objval)

    ###  TEST  ###
    # 1. data
    p_test, pet_test, t_test, qobs_test = test_x[:, 0], test_x[:, 1], test_x[:, 2], test_y[:, 0]
    # 2. initialise the model by loading its climate inputs
    model = get_model(model_name, p_test, pet_test, t_test)
    # 3. run the optimised model for test period
    qsim = model.simulate(paramsmax)
    nse = ObjectiveFunction.nashsutcliffe(qobs_test, qsim)
    print('Test run NSE value = ', nse)

    ### SUMMARY METRICS
    store_results(basin_id, qsim, qobs_test, test_set, save_root, warmup)


if __name__ == '__main__':
    # config
    working_path = '..'
    data_root_path = '../../../CAMELS_US/'
    # basin list
    basin_ids = pd.read_csv(
        os.path.join(data_root_path, 'basin_list.txt'), sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str}
    )['BASIN_ID'].values

    # model name and dataset_bounds
    model_name = 'exp-hydro'
    dataset_bounds = {'train': {'start': '1980-10-01', 'end': '2000-09-30'},
                      'test': {'start': '1999-10-01', 'end': '2010-09-30'}}

    # loop through basin ids
    for i in range(0, len(basin_ids)):  # len(basin_ids)
        basin_id = basin_ids[i]
        try:
            calibrate_test(working_path, data_root_path, model_name, basin_id, dataset_bounds, n_part=15)
        except:
            print('nan found...')
            save_root = f'{working_path}/results/{model_name}/{basin_id}'
            save_path = f'{save_root}/nan_found.xlsx'
            pd.DataFrame({'nan': ['found']}).to_excel(save_path)
