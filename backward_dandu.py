import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. 加载 CSV 数据
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
    return X, X_raw  # 返回归一化后的 X 和原始的 X_raw

# 2. 逆向神经网络
class InverseMaterialNet(nn.Module):
    def __init__(self):
        super(InverseMaterialNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出 (G, K)
        )
    def forward(self, x):
        return self.net(x)

# 3. 训练模型
def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, save_path="results/best_model.pth", save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(save_dir, f"best_model_{timestamp}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"🔥 New best model saved at {model_save_path} with Val Loss: {best_val_loss:.6f}")
    plt.figure(figsize=(18, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", linestyle="-", linewidth=2, marker="o", markevery=5)
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", linestyle="-", linewidth=2, marker="s", markevery=5)
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Curve")
    plt.grid(True)
    loss_curve_path = os.path.join(save_dir, f"loss_curve_{timestamp}.pdf")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"📊 Loss curve saved to {loss_curve_path}")
    return train_losses, val_losses

# 4. 可视化训练损失
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.show()

# 5. 预测并可视化结果
# ... (之前的导入和函数定义保持不变)

# 5. 预测并可视化结果
def visualize_results(model, X_test, X_raw, X_scaler, k_scaler, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_test = X_test.to(device)
    model.eval()
    with torch.no_grad():
        k_pred = model(X_test).cpu().numpy()
    k_pred_real = k_scaler.inverse_transform(k_pred)
    G_real, K_real = 7.35e10, 1.28e11
    G_pred_mean = np.mean(k_pred_real[:, 0])
    K_pred_mean = np.mean(k_pred_real[:, 1])
    G_error_mean = np.abs((G_pred_mean - G_real) / G_real) * 100
    K_error_mean = np.abs((K_pred_mean - K_real) / K_real) * 100
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(8, 5))
    plt.scatter(k_pred_real[:, 0], k_pred_real[:, 1], color="blue", alpha=0.5, label="Predicted (G, K)")
    plt.scatter(G_real, K_real, color="red", marker="*", s=200, label="True (G, K)")
    plt.xlabel("Shear Modulus G (Pa)")
    plt.ylabel("Bulk Modulus K (Pa)")
    plt.title("Predicted vs. True Material Parameters")
    plt.legend()
    plt.grid(True)
    prediction_path = os.path.join(save_dir, f"prediction_results_{timestamp}.pdf")
    plt.savefig(prediction_path, dpi=300)
    plt.close()
    print(f"📊 Prediction results saved to {prediction_path}")
    error_report_path = os.path.join(save_dir, f"error_report_{timestamp}.txt")
    with open(error_report_path, "w") as f:
        f.write(f"📢 Predicted G Mean: {G_pred_mean:.2e}, Error (vs. True G): {G_error_mean:.2f}%\n")
        f.write(f"📢 Predicted K Mean: {K_pred_mean:.2e}, Error (vs. True K): {K_error_mean:.2f}%\n")
        f.write(f"📢 True G: {G_real:.2e}, True K: {K_real:.2e}\n")
    print(f"📄 Error report saved to {error_report_path}")
    save_prediction_details(X_raw, k_pred_real, save_dir=save_dir)
    return k_pred_real, G_error_mean, K_error_mean

# 新增函数：保存预测详情（不含误差数组）
def save_prediction_details(X_raw, k_pred_real, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data = {
        "x-coordinate [mm]": X_raw[:, 0],
        "y-coordinate [mm]": X_raw[:, 1],
        "Predicted_G": k_pred_real[:, 0],
        "Predicted_K": k_pred_real[:, 1]
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(save_dir, f"prediction_details_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 Prediction details saved to {csv_path}")

def main():
    train_csv_path = "missed_data_random10000.csv"
    test_csv_path = "Exp_data/20231116_displacements_raw.csv"
    save_dir = "results"
    X, k, X_scaler, k_scaler = load_fem_data(train_csv_path)
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    k_train, k_val = k[:train_size], k[train_size:]
    train_dataset = TensorDataset(X_train, k_train)
    val_dataset = TensorDataset(X_val, k_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = InverseMaterialNet()
    train_model(model, train_loader, val_loader, epochs=500, lr=0.001, save_dir=save_dir)
    X_test, X_raw = load_test_data(test_csv_path, X_scaler)
    k_pred_real, G_error_mean, K_error_mean = visualize_results(model, X_test, X_raw, X_scaler, k_scaler, save_dir=save_dir)
    print(f"📢 Predicted G Mean: {np.mean(k_pred_real[:, 0]):.2e}, Error (vs. True G): {G_error_mean:.2f}%")
    print(f"📢 Predicted K Mean: {np.mean(k_pred_real[:, 1]):.2e}, Error (vs. True K): {K_error_mean:.2f}%")

if __name__ == "__main__":
    main()



# # 新增函数：运行多次试验
# def run_multiple_experiments(test_csv_path, save_dir="results"):
#     os.makedirs(save_dir, exist_ok=True)
#     results = []
#     for i in range(1, 11):
#         train_csv_path = f"missed_data_random1000_{i}.csv"
#         print(f"\nRunning experiment {i}/10 with {train_csv_path}")
#         X, k, X_scaler, k_scaler = load_fem_data(train_csv_path)
#         train_size = int(0.8 * len(X))
#         X_train, X_val = X[:train_size], X[train_size:]
#         k_train, k_val = k[:train_size], k[train_size:]
#         train_dataset = TensorDataset(X_train, k_train)
#         val_dataset = TensorDataset(X_val, k_val)
#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#         model = InverseMaterialNet()
#         train_model(model, train_loader, val_loader, epochs=1000, lr=0.001, save_dir=save_dir)
#         X_test, X_raw = load_test_data(test_csv_path, X_scaler)
#         k_pred_real, G_error_mean, K_error_mean = visualize_results(model, X_test, X_raw, X_scaler, k_scaler, save_dir=save_dir)
#         G_pred_mean = np.mean(k_pred_real[:, 0])
#         K_pred_mean = np.mean(k_pred_real[:, 1])
#         results.append({
#             "Experiment": i,
#             "Predicted G Mean": G_pred_mean,
#             "Error (vs. True G)": G_error_mean,
#             "Predicted K Mean": K_pred_mean,
#             "Error (vs. True K)": K_error_mean
#         })
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     output_path = os.path.join(save_dir, f"multiple_experiments_results_{timestamp}.txt")
#     with open(output_path, "w") as f:
#         f.write("Multiple Experiments Results\n")
#         f.write("===========================\n\n")
#         f.write("Experiment | Predicted G Mean (Pa) | Error (vs. True G) (%) | Predicted K Mean (Pa) | Error (vs. True K) (%)\n")
#         f.write("-" * 90 + "\n")
#         for result in results:
#             f.write(f"{result['Experiment']:9d} | {result['Predicted G Mean']:12.2e} | {result['Error (vs. True G)']:19.2f} | {result['Predicted K Mean']:17.2e} | {result['Error (vs. True K)']:22.2f}\n")
#     print(f"📄 Multiple experiments results saved to {output_path}")
#     avg_G_mean = np.mean([r["Predicted G Mean"] for r in results])
#     avg_K_mean = np.mean([r["Predicted K Mean"] for r in results])
#     avg_G_error = np.mean([r["Error (vs. True G)"] for r in results])
#     avg_K_error = np.mean([r["Error (vs. True K)"] for r in results])
#     print(f"\nAverage Results across 10 experiments:")
#     print(f"📢 Average Predicted G Mean: {avg_G_mean:.2e}, Average Error (vs. True G): {avg_G_error:.2f}%")
#     print(f"📢 Average Predicted K Mean: {avg_K_mean:.2e}, Average Error (vs. True K): {avg_K_error:.2f}%")

# def main():
#     test_csv_path = "Exp_data/20231116_displacements_raw.csv"
#     save_dir = "results"
#     run_multiple_experiments(test_csv_path, save_dir=save_dir)

# if __name__ == "__main__":
#     main()