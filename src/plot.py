import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from src.helpers import weighted_interval_score

def compare_ensemble(model, model_data, covidhub_data, locations_test, alphas_eval, test_start, test_end,
                     locations_plot = None, location_names = None, pred_type = '1_wk_inc_death', pred_days = 7,
                     saveplots = True, figsize = (6,6)):
    if not locations_plot:
        locations_plot = locations_test
    if not location_names:
        location_names = locations_plot
    if len(locations_plot) != len(location_names):
        raise ValueError('Plotting locations and location names are not of equal length.')

    for i, (query, key, value, y) in enumerate(model_data.get_dataloader(False)):
        if i == 0:
            queries_val = query
            keys_val = key
            values_val = value
            y_val = y
        else:
            queries_val = torch.cat((queries_val, query), dim=0)
            keys_val = torch.cat((keys_val, key), dim=0)
            values_val = torch.cat((values_val, value), dim=0)
            y_val = torch.cat((y_val, y), dim=0)

    if (y_val.size(0) / len(locations_test)) % 1 > 0.:
        raise ValueError('Testing data is not evenly split between locations.')

    for i, (query, key, value, y) in enumerate(covidhub_data.get_dataloader(False)):
        # print(value.size())
        if i == 0:
            values_ensemble = value
            y_ensemble = y
        else:
            values_ensemble = torch.cat((values_ensemble, value), dim=0)
            y_ensemble = torch.cat((y_ensemble, y), dim=0)

    model_out = model.forward(queries_val, keys_val, values_val).detach()
    #attention_weights = model.attention_weights.detach()
    wis_ensemble = weighted_interval_score(values_ensemble, y_val, alphas_eval)
    wis_model = weighted_interval_score(model_out, y_val, alphas_eval)

    num_dates = model
    location_idxs = [locations_test.index(number) for number in locations_plot]
    for i, location in zip(location_idxs, location_names):
        fig = plt.figure(figsize = figsize)
        plt.plot(wis_ensemble[i * num_dates:(i + 1) * num_dates, 0, 0],
                 label='COVIDhub-ensemble, Mean = %0.2f' % wis_ensemble[i * num_dates:(i + 1) * num_dates, 0, 0].mean())
        plt.plot(wis_model[i * num_dates:(i + 1) * num_dates, 0, 0],
                 label='Denpool, Mean = %0.2f' % wis_model[i * num_dates:(i + 1) * num_dates, 0, 0].mean())
        plt.legend()
        plt.ylabel('WIS')
        plt.xlabel('Weeks from %s' % test_start)
        plt.title('%s %s' % (location, pred_type))
        if saveplots:
            plt.savefig('%s_%s_compare_start_%s_end_%s.png' % (pred_type, location, test_start, test_end),
                        dpi = 400, bbox_inches='tight')
        plt.show()