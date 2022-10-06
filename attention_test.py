from src.classes import *
torch.set_default_dtype(torch.float64)

feature_size, num_hidden, dropout, lr, epochs = 3, 20, 0.0, 1e-3, 100
true_model  = LorenzModelPeriodicRho(period = np.inf)
model_zoo   = [LorenzModel(rho=rho) for rho in np.arange(28,50,2)]
lorenz_data = LorenzPeriodicRhoData(true_model, model_zoo)
attention   = AdditiveAttention(feature_size, num_hidden, dropout, lr)

trainer = d2l.Trainer(max_epochs = epochs)
trainer.fit(attention, lorenz_data)

d2l.show_heatmaps([[attention.attention_weights.squeeze(1)]],
                  xlabel='Model One-step Forecasts',
                  ylabel='Batch Input')