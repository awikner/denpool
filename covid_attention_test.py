from src.classes import *
from src.helpers import *
import h5py
import pandas as pd
import timeit
from matplotlib import pyplot as plt

# Covid Data Parameters

truth_path = 'C:\\Users\\user\\Documents\\covid-data\\covid_truth_data.hdf5'
ensemble_path = 'C:\\Users\\user\\Documents\\covid-data\\covid_prediction_data.hdf5'
ensemble_models = ['IHME-CurveFit', 'CU-scenario_mid', 'MOBS-GLEAM_COVID', 'CU-select', 'CU-scenario_low',
                  'CU-nochange', 'GT-DeepCOVID', 'PSI-DRAFT', 'USC-SI_kJalpha']
alphas = ['point', '0.02', '0.05', '0.1', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
prediction_type = ['1_wk_ahead_inc_death', 7]
truth_type = 'Incident Deaths'
start_date = '20-05-09'
end_date = '22-11-05'
train_locations =  ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', '17',
              '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
              '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44',
              '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56']
test_locations = ['11', '24', '51']
time_delay = 5

covid_data = torch.load('covid_dataloader_td5.pth')

output_size = 1+2*(len(alphas)-1)
query_size, key_size, num_hidden, dropout, lr, epochs = time_delay, output_size*time_delay, 1000, 0.0, 5e-5, 501
alphas = torch.tensor([float(alpha) for alpha in covid_data.alphas_eval[1:]], requires_grad=False).reshape(1,1,-1)
attention   = CovidAdditiveAttention(query_size, key_size, num_hidden, dropout, lr, alphas, l1_reg = 0.)
attention.device = "cpu"
trainer = CovidTrainer(max_epochs = epochs)

trainer.fit(attention, covid_data)