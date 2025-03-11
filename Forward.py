import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt


# 1. 加载 CSV 数据
def load_fem_data(csv_path):
    df = pd.read_csv(csv_path)
    X = torch.tensor(df[['x-coordinate [mm]', 'y-coordinate [mm]','G', 'K']].values, dtype=torch.float32)  # 坐标 (x, y)
    y = torch.tensor(df[['x-displacement [mm]', 'y-displacement [mm]']].values, dtype=torch.float32)  # 位移 (u_x, u_y)
    G = torch.tensor(df['G'].values, dtype=torch.float32)  # 剪切模量
    K = torch.tensor(df['K'].values, dtype=torch.float32)  # 体视模量
    return X, y, G, K

# 2. 物理损失函数
import torch

def physics_loss(model, X, G, K):
    """
    计算物理损失，包括：
    - 动量平衡方程：∇·σ = 0
    - 本构方程：σ = λ tr(ε) I + 2μ ε
    - 运动学方程：ε = ∇u
    - 物理损失 = residual_material^2 + residual_balance^2
    
    参数：
    - model: PyTorch 神经网络模型，输入 X，输出位移场 u
    - X: 训练输入坐标 (batch_size, 2)
    - G: 剪切模量 (batch_size,)
    - K: 体积模量 (batch_size,)
    """

    X.requires_grad = True  # 需要计算梯度
    u = model(X)  # 计算位移场 u = [u_x, u_y]
    u_x, u_y = u[:, 0], u[:, 1]  # 分离 u_x 和 u_y

    # **计算应变张量 ε**
    grad_u_x = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    grad_u_y = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

    epsilon_xx = grad_u_x[:, 0]  # ∂u_x/∂x
    epsilon_yy = grad_u_y[:, 1]  # ∂u_y/∂y
    epsilon_xy = 0.5 * (grad_u_x[:, 1] + grad_u_y[:, 0])  # ∂u_x/∂y + ∂u_y/∂x

    # **计算应力张量 σ**
    mu = G  # 剪切模量
    lambda_ = K - (2.0 / 3.0) * mu  # 计算 λ

    trace_epsilon = epsilon_xx + epsilon_yy  # tr(ε)
    sigma_xx = lambda_ * trace_epsilon + 2 * mu * epsilon_xx
    sigma_yy = lambda_ * trace_epsilon + 2 * mu * epsilon_yy
    sigma_xy = 2 * mu * epsilon_xy

    # **计算应力散度 ∇·σ**
    grad_sigma_xx = torch.autograd.grad(sigma_xx, X, grad_outputs=torch.ones_like(sigma_xx), create_graph=True, retain_graph=True)[0][:, 0]
    grad_sigma_yy = torch.autograd.grad(sigma_yy, X, grad_outputs=torch.ones_like(sigma_yy), create_graph=True, retain_graph=True)[0][:, 1]
    grad_sigma_xy_x = torch.autograd.grad(sigma_xy, X, grad_outputs=torch.ones_like(sigma_xy), create_graph=True, retain_graph=True)[0][:, 0]
    grad_sigma_xy_y = torch.autograd.grad(sigma_xy, X, grad_outputs=torch.ones_like(sigma_xy), create_graph=True, retain_graph=True)[0][:, 1]

    # **动量平衡方程残差**
    residual_balance_x = grad_sigma_xx + grad_sigma_xy_y  # ∂σ_xx/∂x + ∂σ_xy/∂y
    residual_balance_y = grad_sigma_xy_x + grad_sigma_yy  # ∂σ_xy/∂x + ∂σ_yy/∂y

    # **本构关系残差**
    residual_material_x = sigma_xx - (lambda_ * trace_epsilon + 2 * mu * epsilon_xx)
    residual_material_y = sigma_yy - (lambda_ * trace_epsilon + 2 * mu * epsilon_yy)

    # **计算总物理损失**
    physics_loss = torch.mean(residual_balance_x**2 + residual_balance_y**2 + residual_material_x**2 + residual_material_y**2)

    return physics_loss


# 3. 边界条件
def dirichlet_bc(model, X_bc, u_bc):
    u_pred = model(X_bc)
    dirichlet_loss = torch.mean((u_pred - u_bc) ** 2)
    return dirichlet_loss

# 4. 训练函数
def train_model(model, X_train, y_train, X_val, y_val, X_left, u_left, X_right, u_right, G, K,
                epochs=200, lr=0.0001, use_physic=False, save_path='models'):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        data_loss = torch.mean((outputs - y_train) ** 2)

        if use_physic:
            phys_loss = physics_loss(model, X_train, G, K)
            left_loss = dirichlet_bc(model, X_left, u_left)
            right_loss = dirichlet_bc(model, X_right, u_right)
            loss = data_loss + phys_loss + left_loss + right_loss
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, "
                  f"data_loss: {data_loss.item():.6f}, physics_loss: {phys_loss.item():.6f}, "
                  f"left_loss: {left_loss.item():.6f}, right_loss: {right_loss.item():.6f}")
        else:
            loss = data_loss
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, data_loss: {data_loss.item():.6f}")

        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = torch.mean((val_outputs - y_val) ** 2)
            print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss.item():.6f}')
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = f"{save_path}/{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Forward Model saved to {save_path}")
        
    print("Training complete.")



# 5. 定义简单的神经网络
class ForwardDisplacementNet(nn.Module):
    def __init__(self):
        super(ForwardDisplacementNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),   # 输入 x, y, G, K
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # 输出 u_x, u_y
        )
    
    def forward(self, x):
        return self.net(x)
    

def plot_displacement(model, X, y):
    device = next(model.parameters()).device  # Get model device
    X = X.to(device)  # Move X to same device as model

    df = pd.DataFrame(X.cpu().numpy(), columns=['x', 'y', 'G', 'K'])  # Convert back to CPU for plotting
    x = df['x']
    y = df['y']

    u_pred = model(X).detach().cpu().numpy()  # Ensure tensor is on CPU before converting to NumPy
    u_x = u_pred[:, 0]
    u_y = u_pred[:, 1]
    u = np.sqrt(u_x ** 2 + u_y ** 2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c=u_x, cmap='coolwarm', s=5)
    plt.colorbar()
    plt.title('Displacement in x-direction')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=u_y, cmap='coolwarm', s=5)
    plt.colorbar()
    plt.title('Displacement in y-direction')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig(f'displacement.pdf')
    plt.close()





def main():
    # 加载数据
    csv_path = "missed_data_random2000.csv"
    X, y, G, K = load_fem_data(csv_path)

    # 分割训练和验证数据
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    G_train, G_val = G[:n_train], G[n_train:]
    K_train, K_val = K[:n_train], K[n_train:]

    # 提取边界条件
    left_mask = (X[:, 0] == X[:, 0].min())  # 左边界 x = min(x)
    right_mask = (X[:, 0] == X[:, 0].max())  # 右边界 x = max(x)
    X_left, u_left = X[left_mask], y[left_mask]
    X_right, u_right = X[right_mask], y[right_mask]

    # 初始化模型
    forward_model = ForwardDisplacementNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    forward_model.to(device)
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_left, u_left = X_left.to(device), u_left.to(device)
    X_right, u_right = X_right.to(device), u_right.to(device)
    G_train, K_train = G_train.to(device), K_train.to(device)

    forward_model_path = 'models/forward_model.pt'
    continue_training = False  # 设置为True，如果你希望在模型存在的情况下继续训练

    if not os.path.exists("models"):
        os.makedirs("models")

    # 训练正向位移模型
    if os.path.exists(forward_model_path) and not continue_training:
        forward_model.load_state_dict(torch.load(forward_model_path))
        print(f"Loaded forward model from {forward_model_path}")
    else:
        print("Training forward model...")
        train_model(forward_model, X_train, y_train, X_val, y_val, X_left, u_left, X_right, u_right, G_train, K_train)
        
    # 绘制位移
    plot_displacement(forward_model, X, y)


if __name__ == "__main__":
    main()
