import os
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import hydroeval as he


def pet_compute(t_mean, dayl):
    # calculate PET using Hamon’s formulation
    pet = 29.8 * (dayl * 24) * 0.611 * np.exp(17.3 * t_mean / (t_mean + 237.3)) / (t_mean + 273.2)
    return pet


class DataBasin():
    def __init__(self, root_path, basin_id):
        self.root_path = root_path
        self.basin_id = basin_id

    def check_validation(self, basin_list, basin_id):
        assert isinstance(basin_id, str), "The basin ID should be a string"
        assert (len(basin_id) == 8 and basin_id.isdigit()), "Basin ID can only be represented by 8 digits"
        assert (basin_id in basin_list.values), "Please confirm the basin specified is in basin_list.txt"

    def load_forcing_data(self, root_path, huc_id, basin_id):
        forcing_path = os.path.join(root_path, 'basin_mean_forcing', 'daymet', huc_id,
                                    basin_id + '_lump_cida_forcing_leap.txt')

        forcing_data = pd.read_csv(forcing_path, sep="\s+|;|:", header=0, skiprows=3, engine='python')
        forcing_data.rename(columns={"Mnth": "Month"}, inplace=True)
        forcing_data['date'] = pd.to_datetime(forcing_data[['Year', 'Month', 'Day']])
        forcing_data['dayl(day)'] = forcing_data['dayl(s)'] / 86400
        forcing_data['tmean(C)'] = (forcing_data['tmin(C)'] + forcing_data['tmax(C)']) / 2

        ## load area from header
        with open(forcing_path, 'r') as fp:
            content = fp.readlines()
            area = int(content[2])

        return forcing_data, area

    def load_flow_data(self, root_path, huc_id, basin_id, area):
        flow_path = os.path.join(root_path, 'usgs_streamflow', huc_id,
                                 basin_id + '_streamflow_qc.txt')

        flow_data = pd.read_csv(flow_path, sep="\s+", names=['Id', 'Year', 'Month', 'Day', 'Q', 'QC'],
                                header=None, engine='python')
        flow_data['date'] = pd.to_datetime(flow_data[['Year', 'Month', 'Day']])
        flow_data['flow(mm)'] = 28316846.592 * flow_data['Q'] * 86400 / (area * 10 ** 6)
        return flow_data

    def load_data(self):
        basin_list = pd.read_csv(os.path.join(self.root_path, 'basin_list.txt'),
                                 sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str})
        self.check_validation(basin_list, self.basin_id)
        huc_id = basin_list[basin_list['BASIN_ID'] == self.basin_id]['HUC'].values[0]
        forcing_data, area = self.load_forcing_data(self.root_path, huc_id, self.basin_id)
        flow_data = self.load_flow_data(self.root_path, huc_id, self.basin_id, area)

        merged_data = pd.merge(forcing_data, flow_data, on='date')
        merged_data = merged_data[(merged_data['date'] >= datetime(1980, 10, 1)) &
                                  (merged_data['date'] <= datetime(2010, 9, 30))]
        merged_data = merged_data.set_index('date')

        # compute pet
        tmean, dayl = merged_data['tmean(C)'].values, merged_data['dayl(day)'].values
        pet = pet_compute(tmean, dayl)

        merged_data['pet(mm/day)'] = pet
        pd_data = merged_data[['prcp(mm/day)', 'pet(mm/day)', 'tmean(C)', 'flow(mm)']]
        print('Data in basin #{} at huc #{} has been successfully loaded.'.format(self.basin_id, huc_id))
        return pd_data


def generate_train_test_conceptual_model(train_set, test_set):
    train_x = train_set.values[:, :-1]
    train_y = train_set.values[:, -1:]
    test_x = test_set.values[:, :-1]
    test_y = test_set.values[:, -1:]
    return train_x, train_y, test_x, test_y


def load_and_process_data(root_path, basin_id, dataset_bounds):
    # dataset bounds
    training_start, training_end = dataset_bounds['train']['start'], dataset_bounds['train']['end']
    testing_start, testing_end = dataset_bounds['test']['start'], dataset_bounds['test']['end']

    # load data
    hydrodata = DataBasin(root_path, basin_id).load_data()

    # split data set to training_set and testing_set
    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]
    print(f"The training data set is from {training_start} to {training_end}, with a shape of {train_set.shape}")
    print(f"The testing data set is from {testing_start} to {testing_end}, with a shape of {test_set.shape}")

    # generate training and testing samples
    train_x, train_y, test_x, test_y = generate_train_test_conceptual_model(train_set, test_set)

    print(f'The shape of train_x, train_y, test_x, and test_y are:')
    print(f'{train_x.shape}, {train_y.shape}, {test_x.shape}, and {test_y.shape}')
    return train_x, train_y, test_x, test_y, test_set


def store_results(basin_id, flow_pred, flow_obs, test_set, save_root, warmup):
    evaluate_set = test_set.loc[:, ['prcp(mm/day)', 'flow(mm)']]
    evaluate_set['flow_obs'] = evaluate_set['flow(mm)']
    evaluate_set['flow_sim'] = np.clip(flow_pred, a_min=0, a_max=None)

    # obs and pred
    y_obs = flow_obs[warmup:]
    y_pred = flow_pred[warmup:]

    # calculate respective NSE and KGE values
    nse = he.evaluator(he.nse, y_pred, y_obs)
    kge, r, _, _ = he.evaluator(he.kge, y_pred, y_obs)
    pbias = he.evaluator(he.pbias, y_pred, y_obs)
    rmse = he.evaluator(he.rmse, y_pred, y_obs)

    # get the min and max values of flow_obs
    min_flow_obs = np.min(y_obs)
    max_flow_obs = np.max(y_obs)

    # compute
    n_rmse = 100 * rmse / (max_flow_obs - min_flow_obs)

    # round
    nse = np.round(nse[0], 3)
    kge, r, pbias = np.round(kge[0], 3), np.round(r[0], 3), np.round(pbias[0], 3)
    n_rmse = np.round(n_rmse[0], 3)

    # stats summary
    stats_summary = pd.DataFrame(data={
        'basin_id': basin_id,
        'nse': [nse],
        'kge': [kge],
        'r': [r],
        'pbias': [pbias],
        'nrmse': n_rmse
    })
    # concatenate
    stats_df = pd.DataFrame(stats_summary)
    # save
    # stats_df.to_excel(f'{save_root}/data.xlsx')

    evaluate_set['basin_id'] = basin_id
    with pd.ExcelWriter(f'{save_root}/data.xlsx') as writer:
        evaluate_set.to_excel(writer, sheet_name='obs_sim')
        stats_df.to_excel(writer, sheet_name='stats')


if __name__ == '__main__':
    # configs
    root_path = '../../../CAMELS_US/'
    basin_id = '01013500'
    dataset_bounds = {'train': {'start': '1980-10-01', 'end': '2000-09-30'},
                      'test': {'start': '1999-10-01', 'end': '2010-09-30'}}

    # train / test data
    train_x, train_y, test_x, test_y, test_set = load_and_process_data(root_path, basin_id, dataset_bounds)
    plt.plot(test_y[:, 0])
    plt.show()
