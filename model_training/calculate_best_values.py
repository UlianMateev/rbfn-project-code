import numpy as np
import os
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq

#Loading the data and parameters
dataset = np.load("processed_data/dataset.npz")
X_train, y_train = dataset['X_train'], dataset['y_train']
X_val, y_val = dataset['X_val'], dataset['y_val']

params = np.load("processed_data/scaling_params.npz")
min_val, max_val = params['min_value'], params['max_value']

P_NEIGHBORS = 15

#Function that reverses the min max normalization
def inverse_transform(scaled_val, min_v, max_v):
    return (scaled_val * (max_v - min_v)) + min_v

y_val_real_mw = inverse_transform(y_val, min_val, max_val)


#K-means calculating function
def get_rbf_centers(X, K, max_iters=50, tolerance=1e-4, batch_size=10000):

    np.random.seed(42)
    centers = X[np.random.choice(len(X), size=K, replace=False)]
    
    for _ in range(max_iters):
        labels = np.zeros(len(X), dtype=int)
        for i in range(0, len(X), batch_size):
            end = min(i + batch_size, len(X))
            batch_dist = cdist(X[i:end], centers)
            labels[i:end] = np.argmin(batch_dist, axis=1)
        
        new_centers = np.array([X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 
                                else X[np.random.choice(len(X))] for j in range(K)])
        
        if np.linalg.norm(new_centers - centers) < tolerance:
            break
        centers = new_centers
        
    return centers


#Calculating the unique width for each center based on nearest neighbour
def calculate_local_sigmas(centers, p_neighbors=P_NEIGHBORS):
    K = len(centers)
    sigmas = np.zeros(K)
    
    distances = cdist(centers, centers)
    
    for i in range(K):
        nearest_distances = np.sort(distances[i])[1 : p_neighbors + 1]
        sigmas[i] = np.mean(nearest_distances)
        if sigmas[i] == 0:
            sigmas[i] = 1e-8
            
    return sigmas


#Calculating the activations of the radial basis functions
def calculate_activations(X, centers, sigmas, batch_size=10000):
    G = np.zeros((len(X), len(centers)))
    
    for i in range(0, len(X), batch_size):
        end = min(i + batch_size, len(X))
        distances = cdist(X[i:end], centers)
        G[i:end] = np.exp(-(distances ** 2) / (2 * sigmas ** 2))
        
    return G


# Calculating the optimal K parameter on the validation set
K_list = [50, 100, 200, 300 500]

best_K = None
lowest_error = float('inf')
best_centers = None
best_sigmas = None


for K in K_list:
    print(f"\nTraining model with K={K}...")
    
    # 1. Finding the centers
    centers = get_rbf_centers(X_train, K)
    
    # 2. Calculating local sigmas
    sigmas = calculate_local_sigmas(centers, p_neighbors=P_NEIGHBORS)
    
    # 3. Training the weights
    G_train = calculate_activations(X_train, centers, sigmas)
    weights, _, _, _ = lstsq(G_train, y_train, lapack_driver='gelsd')
    
    # 4. Test on validation set
    G_val = calculate_activations(X_val, centers, sigmas)
    y_val_pred_scaled = G_val.dot(weights)
    
    # 5. Calculating the real world value error
    y_val_pred_real_mw = inverse_transform(y_val_pred_scaled, min_val, max_val)
    mae = np.mean(np.abs(y_val_real_mw - y_val_pred_real_mw))
    
    print(f"Result for K={K}: Average Error of {mae:.2f} Megawatts")
    
    # 6. Tracking the the values
    if mae < lowest_error:
        lowest_error = mae
        best_K = K
        best_centers = centers
        best_sigmas = sigmas

#Saving the best values
print(f"Best value: K = {best_K} with {lowest_error:.2f} MW error")

os.makedirs("processed_data", exist_ok=True)
np.save("processed_data/best_centers.npy", best_centers)
np.save("processed_data/best_sigmas.npy", best_sigmas)

print(f"Saved {best_K}")