import re
import matplotlib.pyplot as plt
import numpy as np

def parse_metrics(log_file):
    """
    Parse training log and extract RMSE, MSE, and MAE values.
    """
    rmse_values = []
    mse_values = []
    mae_values = []
    epochs = []
    
    # Pattern to match validation metrics
    pattern = r'\[ (\d+) epochs \]valid_rmse:([0-9.]+)\|valid_mse:([0-9.]+)\|valid_mae:([0-9.]+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                rmse = float(match.group(2))
                mse = float(match.group(3))
                mae = float(match.group(4))
                
                epochs.append(epoch)
                rmse_values.append(rmse)
                mse_values.append(mse)
                mae_values.append(mae)
    
    return epochs, rmse_values, mse_values, mae_values

def plot_metrics(epochs, rmse_values, mse_values, mae_values):
    """
    Plot the three metrics over epochs.
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, rmse_values, 'b-', label='RMSE')
    plt.plot(epochs, mse_values, 'r-', label='MSE')
    plt.plot(epochs, mae_values, 'g-', label='MAE')
    
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Validation Metrics Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add final values to legend
    plt.legend([f'RMSE (final={rmse_values[-1]:.4f})',
               f'MSE (final={mse_values[-1]:.4f})',
               f'MAE (final={mae_values[-1]:.4f})'])
    
    plt.tight_layout()
    plt.savefig('nre_validation_metrics.png')
    plt.close()

# Usage
epochs, rmse, mse, mae = parse_metrics('training_log_nre_new.txt')
plot_metrics(epochs, rmse, mse, mae)