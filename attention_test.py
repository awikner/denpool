from src.classes import *

true_model = LorenzModelPeriodicRho()
model_zoo = [LorenzModel(rho = rho) for rho in np.arange(28,50,2)]
lorenz_data = LorenzPeriodicRhoData(true_model, model_zoo)
attention = AdditiveAttention(num_hiddens = 20, dropout = 0.0)
keys, values, queries, ys = next(iter(lorenz_data.train_dataloader()))
print(keys.size())
print(values.size())
print(queries.size())
print(ys.size())
a_out = attention(queries, keys, values)
print(a_out.size())