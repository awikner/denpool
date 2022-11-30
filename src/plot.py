import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from src.helpers import weighted_interval_score

def compare_ensemble(models, model_data, covidhub_data, locations_test, alphas_eval, test_start, test_end,
                     locations_plot = None, location_names = None,
                     pred_type = '1_wk_inc_death', pred_days = 7,
                     saveplots = True, figsize = (6,6), model_names = None,
                     log = False):
    if not locations_plot:
        locations_plot = locations_test
    if not location_names:
        location_names = locations_plot
    if len(locations_plot) != len(location_names):
        raise ValueError('Plotting locations and location names are not of equal length.')
    if not model_names:
        model_names = ['Model %d' % (i+1) for i in range(len(models))]

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

    tests_per_loc = (y_val.size(0) / len(locations_test))
    if (y_val.size(0) / len(locations_test)) % 1 > 0.:
        raise ValueError('Testing data is not evenly split between locations.')
    tests_per_loc = int(tests_per_loc)

    for i, (query, key, value, y) in enumerate(covidhub_data.get_dataloader(False)):
        # print(value.size())
        if i == 0:
            values_ensemble = value
            y_ensemble = y
        else:
            values_ensemble = torch.cat((values_ensemble, value), dim=0)
            y_ensemble = torch.cat((y_ensemble, y), dim=0)

    models_out = [model.forward(queries_val, keys_val, values_val).detach() for model in models]
    if hasattr(model_data, 'rescale_factors'):
        for l, loc in enumerate(locations_test):
            for j in range(len(models)):
                models_out[j][l * tests_per_loc:(l + 1) * tests_per_loc, :, :] *= model_data.rescale_factors[loc]
            y_val[l * tests_per_loc:(l + 1) * tests_per_loc, :, :] *= model_data.rescale_factors[loc]
    #attention_weights = model.attention_weights.detach()
    wis_ensemble = weighted_interval_score(values_ensemble, y_val, alphas_eval)
    wis_models = [weighted_interval_score(model_out, y_val, alphas_eval) for model_out in models_out]
    if log:
        mean_str = 'Log WIS'
        wis_ensemble = torch.log(wis_ensemble)
        wis_models   = [torch.log(wis_model) for wis_model in wis_models]
    else:
        mean_str = 'WIS'
    mean_wis_ensemble = wis_ensemble.mean()
    mean_wis_models   = [wis_model.mean() for wis_model in wis_models]
    print('Ensemble Mean WIS: %f' % mean_wis_ensemble)
    for model_name, mean_wis in zip(model_names, mean_wis_models):
        print('%s Mean WIS: %f' % (model_name, mean_wis))

    num_dates = tests_per_loc
    location_idxs = [locations_test.index(number) for number in locations_plot]
    markers = ['--', '--', '--']
    for i, location in zip(location_idxs, location_names):
        fig, ax1 = plt.subplots(figsize = figsize)
        ax2 = ax1.twinx()
        ax1.plot(wis_ensemble[i * num_dates:(i + 1) * num_dates, 0, 0], 'o-',
                 label='Mean = %0.2f' % (wis_ensemble[i * num_dates:(i + 1) * num_dates, 0, 0].mean()))
        for j, (wis_model, model_name) in enumerate(zip(wis_models, model_names)):
            ax1.plot(wis_model[i * num_dates:(i + 1) * num_dates, 0, 0], markers[j],
                     label='Mean = %0.2f' % (wis_model[i * num_dates:(i + 1) * num_dates, 0, 0].mean()))

        #ax1.plot(wis_ensemble[i * num_dates:(i + 1) * num_dates, 0, 0],'o-',
        #         label='COVIDhub-ensemble')
        #for j, (wis_model, model_name) in enumerate(zip(wis_models, model_names)):
        #    ax1.plot(wis_model[i * num_dates:(i + 1) * num_dates, 0, 0], markers[j],
        #             label='%s' % model_name)
        ax2.plot(y_val[i*num_dates: (i+1) * num_dates, 0, 0], 'x:',  label = 'Ground Truth')
        _, x_max = ax1.get_xlim()
        #print(scaled_ground_truth)
        #plt.plot(scaled_ground_truth, '--', label = 'Ground Truth (Scaled)')
        ax1.legend(loc = 'best')
        #ax1.legend(ncol = 2)
        #ax2.legend(loc = 'lower left')
        ax1.set_ylabel(mean_str)
        ax2.set_ylabel(pred_type)
        ax1.set_xlabel('Weeks from %s' % test_start)
        plt.title('%s %s' % (location, pred_type))
        plt.xticks(np.arange(0,x_max,2))
        if saveplots:
            plt.savefig('%s_%s_compare_start_%s_end_%s_legend.pdf' % (pred_type, location, test_start, test_end),
                        dpi = 400, bbox_inches='tight')
        plt.show()

def compare_wis_cdf(models, model_data, covidhub_data, locations_test, alphas_eval, test_start, test_end,
                     pred_type = '1_wk_inc_death',
                     saveplots = True, figsize = (6,6), model_names = None,
                     log = False):

    if not model_names:
        model_names = ['Model %d' % (i+1) for i in range(len(models))]

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

    tests_per_loc = (y_val.size(0) / len(locations_test))
    if (y_val.size(0) / len(locations_test)) % 1 > 0.:
        raise ValueError('Testing data is not evenly split between locations.')
    tests_per_loc = int(tests_per_loc)

    for i, (query, key, value, y) in enumerate(covidhub_data.get_dataloader(False)):
        # print(value.size())
        if i == 0:
            values_ensemble = value
            y_ensemble = y
        else:
            values_ensemble = torch.cat((values_ensemble, value), dim=0)
            y_ensemble = torch.cat((y_ensemble, y), dim=0)

    models_out = [model.forward(queries_val, keys_val, values_val).detach() for model in models]
    if hasattr(model_data, 'rescale_factors'):
        for l, loc in enumerate(locations_test):
            for j in range(len(models)):
                models_out[j][l * tests_per_loc:(l + 1) * tests_per_loc, :, :] *= model_data.rescale_factors[loc]
            y_val[l * tests_per_loc:(l + 1) * tests_per_loc, :, :] *= model_data.rescale_factors[loc]
    #attention_weights = model.attention_weights.detach()
    wis_ensemble = weighted_interval_score(values_ensemble, y_val, alphas_eval)
    wis_models = [weighted_interval_score(model_out, y_val, alphas_eval) for model_out in models_out]
    if log:
        mean_str = 'Log WIS'
        wis_ensemble = torch.log(wis_ensemble)
        wis_models   = [torch.log(wis_model) for wis_model in wis_models]
    else:
        mean_str = 'WIS'

    ensemble_wis_cdf = np.sort(wis_ensemble[:,0,0].numpy())
    models_wis_cdf = [np.sort(wis_model[:,0,0].numpy()) for wis_model in wis_models]




    num_dates = tests_per_loc
    fig = plt.figure(figsize = figsize)
    plt.semilogx(ensemble_wis_cdf, np.cumsum(np.ones(ensemble_wis_cdf.size))/ensemble_wis_cdf.size,
             label='COVIDhub-ensemble')
    for model_wis_cdf, model_name in zip(models_wis_cdf, model_names):
        plt.semilogx(model_wis_cdf, np.cumsum(np.ones(model_wis_cdf.size))/model_wis_cdf.size,
             label='%s' % (model_name))
    #_, y_max = fig.ax.get_ylim()
    #scaled_ground_truth = y_val[i*num_dates: (i+1 * num_dates), 0, 0] *\
    #                      0.9*y_max/torch.max(y_val[i*num_dates: (i+1 * num_dates), 0, 0])
    #plt.plot(scaled_ground_truth, '--', label = 'Ground Truth')
    plt.legend()
    plt.xlabel(mean_str)
    plt.xlabel('CDF from %s' % test_start)
    plt.title('%s' % (pred_type))
    if saveplots:
        plt.savefig('%s_cdf_compare_start_%s_end_%s.png' % (pred_type, test_start, test_end),
                    dpi = 400, bbox_inches='tight')
    plt.show()