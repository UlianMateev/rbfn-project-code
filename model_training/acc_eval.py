import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

#Lodaing the test data and model
print("Loading unseen Test Data and final Model")

#Unseen test data
dataset = np.load("processed_data/dataset.npz")
X_test = dataset['X_test']
y_test = dataset['y_test']


model = np.load("processed_data/rbf_model_opt.npz")
centers = model['centers']
sigmas = model['sigma']  # This is now your array of Local Sigmas!
weights = model['weights']


params = np.load("processed_data/scaling_params.npz")
min_val = params['min_value']
max_val = params['max_value']

print(f"Loaded {len(X_test)} hours of test data with {X_test.shape[1]} features each.")


#Function that calculates the radial basis activations
def calculate_activations(X, centers, sigmas, batch_size=10000):
    G = np.zeros((len(X), len(centers)))
    for i in range(0, len(X), batch_size):
        end = min(i + batch_size, len(X))
        distances = cdist(X[i:end], centers)
        G[i:end] = np.exp(-(distances ** 2) / (2 * sigmas ** 2))
    return G


#Reversing the min max normalization
def inverse_transform(scaled_val, min_v, max_v):
    return (scaled_val * (max_v - min_v)) + min_v

print("Running predictions on the Test Set...")

#Calculating the hidden layer activations
G_test = calculate_activations(X_test, centers, sigmas)

#Multiplying by the weights to get the predicted value
y_pred_scaled = G_test.dot(weights)

#Converting the model prediction and real test values into MW (Reversing the normalization)
y_pred_real_mw = inverse_transform(y_pred_scaled, min_val, max_val)
y_test_real_mw = inverse_transform(y_test, min_val, max_val)


#Calculating evaluation metrics
errors = y_test_real_mw - y_pred_real_mw
abs_errors = np.abs(errors)

#Standard errors
mae = np.mean(abs_errors)
rmse = np.sqrt(np.mean(errors ** 2))

#Percentage error (Added a tiny epsilon to prevent divide-by-zero just in case)
mape = np.mean(abs_errors / (y_test_real_mw + 1e-8)) * 100

#R-squared Score
ss_res = np.sum(errors ** 2)
ss_tot = np.sum((y_test_real_mw - np.mean(y_test_real_mw)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

#Outlier Analysis
max_error = np.max(abs_errors)
medae = np.median(abs_errors)

print("==========================================")
print(f"R-Squared (R²):          {r2_score:.4f}")
print(f"Average Error (MAE):     ±{mae:.2f} MW")
print(f"Median Error (MedAE):    ±{medae:.2f} MW")
print(f"Worst-Case Error (RMSE): ±{rmse:.2f} MW")
print(f"Max Single Miss:         ±{max_error:.2f} MW")
print(f"Percentage Error (MAPE): {mape:.2f}%")
print("==========================================")

#Ploting 2 weeks of data for visualization
print("\nGenerating graph for documentation")


hours_to_plot = 336  #2 weeks in hours

plt.figure(figsize=(14, 6))


plt.plot(y_test_real_mw[:hours_to_plot], label='Реално Натоварване', color='black', linewidth=1.5)


plt.plot(y_pred_real_mw[:hours_to_plot], label='RBFN Предвидени', color='red', linestyle='--', linewidth=1.5)

plt.title('RBFN: Реални срещу Предвидени стойности в мегаватове (2-седмици)', fontsize=14, fontweight='bold')
plt.xlabel('Time (Hours)', fontsize=12)
plt.ylabel('Energy Load (Megawatts)', fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()