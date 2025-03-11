import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from tqdm import tqdm
from itertools import product

# 1. 数据加载
def load_fem_data(csv_path):
    df = pd.read_csv(csv_path)
    X_raw = df[['x-coordinate [mm]', 'y-coordinate [mm]', 'x-displacement [mm]', 'y-displacement [mm]']].values
    k_raw = df[['G', 'K']].values
    X_scaler = StandardScaler()
    k_scaler = StandardScaler()
    X = torch.tensor(X_scaler.fit_transform(X_raw), dtype=torch.float32)
    k = torch.tensor(k_scaler.fit_transform(k_raw), dtype=torch.float32)
    return X, k, X_scaler, k_scaler

def load_test_data(test_csv_path, X_scaler):
    df = pd.read_csv(test_csv_path)
    X_raw = df[['x-coordinate [mm]', 'y-coordinate [mm]', 'x-displacement [mm]', 'y-displacement [mm]']].values
    X = torch.tensor(X_scaler.transform(X_raw), dtype=torch.float32)
    return X

# 2. 评估指标类
class EvaluationMetrics:
    def __init__(self):
        self.np = np
        self.torch = torch
        self.mse_func = mean_squared_error
        self.mae_func = mean_absolute_error
        self.r2_func = r2_score
    
    def _ensure_numpy(self, tensor):
        if isinstance(tensor, self.torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def mse(self, y_true, y_pred):
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)
        return self.mse_func(y_true, y_pred)
    
    def rmse(self, y_true, y_pred):
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)
        return self.np.sqrt(self.mse_func(y_true, y_pred))
    
    def mae(self, y_true, y_pred):
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)
        return self.mae_func(y_true, y_pred)
    
    def mape(self, y_true, y_pred):
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)
        return self.np.mean(self.np.abs((y_true - y_pred) / y_true)) * 100
    
    def r2(self, y_true, y_pred):
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)
        return self.r2_func(y_true, y_pred)
    
    def relative_error(self, y_true, y_pred):
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)
        return self.np.abs((y_true - y_pred) / y_true) * 100
    
    def evaluate_all(self, y_true, y_pred):
        results = {
            "MSE": self.mse(y_true, y_pred),
            "RMSE": self.rmse(y_true, y_pred),
            "MAE": self.mae(y_true, y_pred),
            "MAPE (%)": self.mape(y_true, y_pred),
            "R²": self.r2(y_true, y_pred),
            "Relative Error (%)": self.np.mean(self.relative_error(y_true, y_pred))
        }
        return results



def evaluate_material_predictions(model, data_loader, k_scaler, true_values=None, device="mps"):
    metrics = EvaluationMetrics()
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for X_batch, k_batch in data_loader:
            X_batch = X_batch.to(device)
            k_batch = k_batch.to(device)
            k_pred = model(X_batch)
            all_preds.append(k_pred.cpu().numpy())
            all_trues.append(k_batch.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    
    all_preds_real = k_scaler.inverse_transform(all_preds)
    all_trues_real = k_scaler.inverse_transform(all_trues)
    
    G_pred = all_preds_real[:, 0]
    K_pred = all_preds_real[:, 1]
    G_true = all_trues_real[:, 0]
    K_true = all_trues_real[:, 1]
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    if true_values is not None:
        G_true_exp, K_true_exp = true_values
        G_true_exp_array = np.full_like(G_pred, G_true_exp)
        K_true_exp_array = np.full_like(K_pred, K_true_exp)
        metrics_G_exp = metrics.evaluate_all(G_true_exp_array, G_pred)
        metrics_K_exp = metrics.evaluate_all(K_true_exp_array, K_pred)
        
        # 打印并保存实验真实值比较结果
        eval_text = "=== 与实验真实值比较 ===\n"
        print("\n=== 与实验真实值比较 ===")
        for metric, value in metrics_G_exp.items():
            print(f"  G {metric}: {value:.6f}")
            eval_text += f"  G {metric}: {value:.6f}\n"
        for metric, value in metrics_K_exp.items():
            print(f"  K {metric}: {value:.6f}")
            eval_text += f"  K {metric}: {value:.6f}\n"
        
        eval_file = os.path.join(save_dir, f"evaluation_results_{timestamp}.txt")
        with open(eval_file, "w") as f:
            f.write(eval_text)
        print(f"\n评估结果已保存至 {eval_file}")
        
        return metrics_G_exp, metrics_K_exp
    
    metrics_G = metrics.evaluate_all(G_true, G_pred)
    metrics_K = metrics.evaluate_all(K_true, K_pred)
    
    # 打印并保存验证集评估结果
    eval_text = "=== 验证集评估 ===\n"
    print("\n=== 验证集评估 ===")
    for metric, value in metrics_G.items():
        print(f"  G {metric}: {value:.6f}")
        eval_text += f"  G {metric}: {value:.6f}\n"
    for metric, value in metrics_K.items():
        print(f"  K {metric}: {value:.6f}")
        eval_text += f"  K {metric}: {value:.6f}\n"
    
    eval_file = os.path.join(save_dir, f"evaluation_results_{timestamp}.txt")
    with open(eval_file, "w") as f:
        f.write(eval_text)
    print(f"\n评估结果已保存至 {eval_file}")
    
    return metrics_G, metrics_K

# 3. 模型创建
def create_model(hidden_sizes, activation='relu', dropout_rate=0.0):
    layers = []
    input_size = 4  # (x, y, u_x, u_y)
    
    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid()
    }
    
    act_func = activations.get(activation.lower(), nn.ReLU())
    
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(act_func)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        input_size = hidden_size
    
    layers.append(nn.Linear(input_size, 2))
    return nn.Sequential(*layers)

# 4. 超参数搜索
# 4. 超参数搜索
def hyperparameter_search(X_dict, k_dict, X_scaler_dict, k_scaler_dict, search_type='grid', n_trials=10, 
                         cv_folds=3, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    if search_type == 'grid':
        param_grid = {
            'hidden_sizes': [
                        [8], [16], [32], [64],               # 1 层隐藏层
                        [8, 16], [16, 32], [32, 64], [64, 128],  # 2 层隐藏层
                        [8, 16, 8], [16, 32, 16], [32, 64, 32], [64, 128, 64]  # 3 层隐藏层
                             ],
            'activation': ['relu'],
            'dropout_rate': [0.0],
            'learning_rate': [0.001],
            'batch_size': [16, 32],
            'sample_size': [1000, 2000, 4000, 10000, 50000]  # 新增样本量参数
        }
        keys = param_grid.keys()
        param_combinations = [dict(zip(keys, values)) for values in product(*param_grid.values())]
    else:
        def sample_params():
            hidden_sizes_options = [
                                    [8], [16], [32], [64],          
                                    [8, 16], [16, 32], [32, 64], [64, 128], 
                                    [8, 16, 8], [16, 32, 16], [32, 64, 32], [64, 128, 64]  
                                    ]
            return {
                'hidden_sizes': np.random.choice(hidden_sizes_options),
                'activation': np.random.choice(['relu']),
                'dropout_rate': np.random.choice([0.0]),
                'learning_rate': 10 ** np.random.uniform(-4, -2),
                'batch_size': np.random.choice([16, 32]),
                'sample_size': np.random.choice([1000, 2000, 4000, 10000, 50000])
            }
        param_combinations = [sample_params() for _ in range(n_trials)]
    
    print("\n=== 即将运行的超参数组合 ===")
    print(f"总共 {len(param_combinations)} 个组合:")
    combinations_text = f"总共 {len(param_combinations)} 个组合:\n"
    for i, params in enumerate(param_combinations, 1):
        print(f"组合 {i}: {params}")
        combinations_text += f"组合 {i}: {params}\n"
    
    combinations_file = os.path.join(save_dir, f"hyperparameter_combinations_{timestamp}.txt")
    with open(combinations_file, "w") as f:
        f.write(combinations_text)
    print(f"\n超参数组合已保存至 {combinations_file}")
    
    while True:
        confirmation = input("\n是否继续运行超参数搜索？(输入 'y' 继续，'n' 退出): ").strip().lower()
        if confirmation == 'y':
            break
        elif confirmation == 'n':
            print("已取消超参数搜索。")
            return None, None, None
        else:
            print("无效输入，请输入 'y' 或 'n'。")
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    results = []
    best_val_loss = float('inf')
    best_model = None
    best_params = None
    
    for trial_num, params in enumerate(tqdm(param_combinations, desc="Hyperparameter Search")):
        print(f"\nStarting trial {trial_num + 1}/{len(param_combinations)} with params: {params}")
        sample_size = params['sample_size']
        X = X_dict[sample_size]
        k = k_dict[sample_size]
        dataset = TensorDataset(X, k)
        
        cv_losses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold + 1}/{cv_folds}")
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=val_sampler)
            
            model = create_model(params['hidden_sizes'], params['activation'], params['dropout_rate'])
            model.to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            for epoch in range(10):
                model.train()
                for X_batch, k_batch in train_loader:
                    X_batch, k_batch = X_batch.to(device), k_batch.to(device)
                    optimizer.zero_grad()
                    predictions = model(X_batch)
                    loss = criterion(predictions, k_batch)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, k_batch in val_loader:
                    X_batch, k_batch = X_batch.to(device), k_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, k_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            cv_losses.append(val_loss)
            print(f"  Fold {fold + 1} validation loss: {val_loss:.6f}")
        
        mean_cv_loss = np.mean(cv_losses)
        print(f"Trial {trial_num + 1} mean CV loss: {mean_cv_loss:.6f}")
        results.append({
            'trial': trial_num + 1,
            'hidden_sizes': str(params['hidden_sizes']),
            'activation': params['activation'],
            'dropout_rate': params['dropout_rate'],
            'learning_rate': params['learning_rate'],
            'batch_size': params['batch_size'],
            'sample_size': params['sample_size'],
            'mean_val_loss': mean_cv_loss
        })
        
        if mean_cv_loss < best_val_loss:
            best_val_loss = mean_cv_loss
            best_model = create_model(params['hidden_sizes'], params['activation'], params['dropout_rate'])
            best_params = params.copy()
    
    results_df = pd.DataFrame(results).sort_values('mean_val_loss')
    results_df.to_csv(os.path.join(save_dir, f"hyperparameter_search_results_{timestamp}.csv"), index=False)
    print(f"\n超参数搜索结果已保存至 {os.path.join(save_dir, f'hyperparameter_search_results_{timestamp}.csv')}")
    
    print("\n=== 超参数组合性能排名 ===")
    ranking_text = "=== 超参数组合性能排名 ===\n"
    for idx, row in results_df.iterrows():
        rank = idx + 1
        params_str = (f"hidden_sizes={row['hidden_sizes']}, activation={row['activation']}, "
                      f"dropout_rate={row['dropout_rate']}, learning_rate={row['learning_rate']}, "
                      f"batch_size={row['batch_size']}, sample_size={row['sample_size']}")
        print(f"Rank {rank}: Trial {int(row['trial'])}, Mean CV Loss: {row['mean_val_loss']:.6f}, Params: {params_str}")
        ranking_text += (f"Rank {rank}: Trial {int(row['trial'])}, Mean CV Loss: {row['mean_val_loss']:.6f}, "
                         f"Params: {params_str}\n")
    
    best_row = results_df.iloc[0]
    print(f"\n🏆 最佳组合 (Rank 1): Trial {int(best_row['trial'])}, Mean CV Loss: {best_row['mean_val_loss']:.6f}")
    print(f"参数: {best_params}")
    ranking_text += (f"\n🏆 最佳组合 (Rank 1): Trial {int(best_row['trial'])}, Mean CV Loss: {best_row['mean_val_loss']:.6f}\n"
                     f"参数: {best_params}\n")
    
    with open(combinations_file, "a") as f:
        f.write("\n" + ranking_text)
    print(f"性能排名已追加至 {combinations_file}")
    
    return best_model, best_params, results_df
    

# 5. 训练模型
def train_model_with_checkpoints(model, train_loader, val_loader, epochs=100, lr=0.001, 
                                save_dir="results", resume=False, checkpoint_freq=10):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    
    # 修改设备检测，支持 MPS
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    start_epoch = 0
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        print(f"✅ 从 epoch {start_epoch} 恢复训练")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for X_batch, k_batch in train_loader:
            X_batch, k_batch = X_batch.to(device), k_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, k_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, k_batch in val_loader:
                X_batch, k_batch = X_batch.to(device), k_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, k_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"best_model_{timestamp}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 新的最佳模型已保存: {best_model_path}")
        
        if (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            print(f"📑 Epoch {epoch+1} 的检查点已保存")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"loss_curve_{timestamp}.png"), dpi=300)
    plt.close()
    return train_losses, val_losses, best_val_loss

# 6. 可视化和预测
def visualize_results(model, X_test, X_scaler, k_scaler, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    # 修改设备检测，支持 MPS
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    X_test = X_test.to(device)
    
    model.eval()
    with torch.no_grad():
        k_pred = model(X_test).cpu().numpy()
    
    k_pred_real = k_scaler.inverse_transform(k_pred)
    G_real, K_real = 7.35e10, 1.28e11
    G_pred, K_pred = k_pred_real[:, 0], k_pred_real[:, 1]
    G_error = np.abs((G_pred - G_real) / G_real) * 100
    K_error = np.abs((K_pred - K_real) / K_real) * 100
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(8, 5))
    plt.scatter(G_pred, K_pred, color="blue", alpha=0.5, label="Predicted (G, K)")
    plt.scatter(G_real, K_real, color="red", marker="*", s=200, label="True (G, K)")
    plt.xlabel("Shear Modulus G (Pa)")
    plt.ylabel("Bulk Modulus K (Pa)")
    plt.title("Predicted vs. True Material Parameters")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"prediction_results_{timestamp}.png"), dpi=300)
    plt.close()
    
    error_report_path = os.path.join(save_dir, f"error_report_{timestamp}.txt")
    with open(error_report_path, "w") as f:
        f.write(f"Predicted G Mean: {np.mean(G_pred):.2e}, Error: {np.mean(G_error):.2f}%\n")
        f.write(f"Predicted K Mean: {np.mean(K_pred):.2e}, Error: {np.mean(K_error):.2f}%\n")
        f.write(f"True G: {G_real:.2e}, True K: {K_real:.2e}\n")
    
    return k_pred_real, G_error, K_error

## 7. 主函数
def main():
    save_dir = "results"
    test_csv_path = "Exp_data/20231116_displacements_raw.csv"
    
    # 定义不同样本量的训练数据路径
    train_paths = {
        1000: "missed_data_random1000.csv",
        2000: "missed_data_random2000.csv",
        4000: "missed_data_random4000.csv",
        10000: "missed_data_random10000.csv",
        50000: "missed_data_random50000.csv",
    }
    
    # 加载所有数据集
    X_dict, k_dict, X_scaler_dict, k_scaler_dict = {}, {}, {}, {}
    for sample_size, path in train_paths.items():
        print(f"Loading data for sample size {sample_size} from {path}")
        X, k, X_scaler, k_scaler = load_fem_data(path)
        X_dict[sample_size] = X
        k_dict[sample_size] = k
        X_scaler_dict[sample_size] = X_scaler
        k_scaler_dict[sample_size] = k_scaler
    
    # 超参数搜索
    best_model, best_params, _ = hyperparameter_search(
        X_dict, k_dict, X_scaler_dict, k_scaler_dict, search_type='grid', save_dir=save_dir
    )
    if best_model is None:
        print("程序已退出。")
        return
    
    print(f"Best parameters: {best_params}")
    
    # 使用最佳样本量的数据进行训练
    best_sample_size = best_params['sample_size']
    X = X_dict[best_sample_size]
    k = k_dict[best_sample_size]
    k_scaler = k_scaler_dict[best_sample_size]
    
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    k_train, k_val = k[:train_size], k[train_size:]
    
    train_dataset = TensorDataset(X_train, k_train)
    val_dataset = TensorDataset(X_val, k_val)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # 训练最佳模型
    train_model_with_checkpoints(best_model, train_loader, val_loader, epochs=200, lr=best_params['learning_rate'], 
                                 save_dir=save_dir, resume=False, checkpoint_freq=10)
    
    # 加载测试数据并预测
    X_test = load_test_data(test_csv_path, X_scaler_dict[best_sample_size])
    k_pred_real, G_error, K_error = visualize_results(best_model, X_test, X_scaler_dict[best_sample_size], 
                                                     k_scaler, save_dir=save_dir)
    
    # 评估模型
    metrics_G, metrics_K = evaluate_material_predictions(best_model, val_loader, k_scaler, 
                                                        true_values=(7.35e10, 1.28e11))
    
    pred_text = f"Predicted G Mean: {np.mean(k_pred_real[:, 0]):.2e}, Error: {np.mean(G_error):.2f}%\n"
    pred_text += f"Predicted K Mean: {np.mean(k_pred_real[:, 1]):.2e}, Error: {np.mean(K_error):.2f}%\n"
    print(pred_text)
    
    eval_file = os.path.join(save_dir, f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(eval_file, "a") as f:
        f.write("\n=== 预测结果 ===\n")
        f.write(pred_text)
    print(f"预测结果已追加至 {eval_file}")

if __name__ == "__main__":
    main()