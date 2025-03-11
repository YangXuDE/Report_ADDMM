import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import argparse
import re
import random

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime



class PINN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(PINN, self).__init__()
        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]  # 输入层
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]  # 隐藏层
        layers.append(nn.Linear(hidden_dim, 2))  # 输出层
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def combine_snapshots_to_csv(snapshot_dir, output_file_base, snapshot_num=None, last_epoch=None, hidden_dim=64, num_layers=4):
    """Combine specified number of snapshot CSV files into one with G and K parameters.
    
    Args:
        snapshot_dir: 快照文件目录
        output_file_base: 输出文件的基础路径（不含时间戳和数量信息）
        snapshot_num: 指定使用的快照数量 None表示使用所有
    """
    if not os.path.exists(snapshot_dir):
        raise FileNotFoundError(f"Directory {snapshot_dir} does not exist")
    
    if not output_file_base or not isinstance(output_file_base, str):
        raise ValueError(f"Invalid output_file_base: '{output_file_base}'. Must be a non-empty string.")

    data_frames = []
    pattern = re.compile(r'snapshot_G([\d\.eE+-]+)_K([\d\.eE+-]+)\.csv')  # 正则匹配文件名
    
    # 获取所有匹配的文件
    matching_files = []
    for filename in os.listdir(snapshot_dir):
        match = pattern.match(filename)
        if match:
            try:
                G, K = map(float, match.groups())  # 解析 G 和 K
                matching_files.append((filename, G, K))
            except Exception as e:
                print(f"Error parsing file {filename}: {e}")
    
    total_files = len(matching_files)
    if not total_files:
        raise ValueError(f"No valid snapshot files found in {snapshot_dir}")
    
    # 验证 snapshot_num
    if snapshot_num is not None:
        if snapshot_num <= 0:
            raise ValueError(f"snapshot_num must be positive, got {snapshot_num}")
        if snapshot_num > total_files:
            print(f"Warning: Requested {snapshot_num} snapshots, but only {total_files} available. Using all {total_files} snapshots.")
            snapshot_num = total_files
        print(f"Selecting {snapshot_num} snapshots from {total_files} available")
        matching_files = random.sample(matching_files, snapshot_num)
    else:
        snapshot_num = total_files
        print(f"No snapshot_num specified, using all {total_files} snapshots")
    
    # 处理选定的文件
    for filename, G, K in matching_files:
        try:
            df = pd.read_csv(os.path.join(snapshot_dir, filename))
            df['G'] = G
            df['K'] = K
            data_frames.append(df)
            print(f"Processed: {filename} (G={G:.2e}, K={K:.2e})")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    if not data_frames:
        raise ValueError(f"No valid snapshot files processed from {snapshot_dir}")

    # 生成带时间戳和数量的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_snapshot_num = len(data_frames)  # 实际使用的 snapshot 数量
    last_epoch = last_epoch if last_epoch is not None else data_frames[0].shape[0] - 1  # 使用第一个文件的最后一个 epoch 数
    output_file = f"{output_file_base}_{timestamp}_sn{actual_snapshot_num}_e{last_epoch}_hd{hidden_dim}_nl{num_layers}.csv"
    
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir:  # 仅在目录非空时创建
        os.makedirs(output_dir, exist_ok=True)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined {actual_snapshot_num} snapshots saved to {output_file}")
    return combined_df, output_file  # 返回 DataFrame 和实际文件名


def load_and_preprocess_data(df, test_size=0.2, random_state=42):
    """Load and preprocess data with train-validation split from combined snapshots.
    
    Args:
        df: Combined DataFrame from snapshots
        test_size: Fraction of data for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (x_train, u_train, x_val, u_val, scalers)
    """
    required_columns = ['x-coordinate [mm]', 'y-coordinate [mm]', 'G', 'K', 
                       'x-displacement [mm]', 'y-displacement [mm]']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"Missing columns in data: {set(required_columns) - set(df.columns)}")

    scalers = {col: MinMaxScaler() for col in required_columns}
    df_scaled = df.copy()
    
    for col in required_columns:
        df_scaled[col] = scalers[col].fit_transform(df[[col]])

    features = df_scaled[['x-coordinate [mm]', 'y-coordinate [mm]', 'G', 'K']].values
    targets = df_scaled[['x-displacement [mm]', 'y-displacement [mm]']].values

    indices = np.random.RandomState(random_state).permutation(len(df))
    split_idx = int(len(df) * (1 - test_size))

    return (
        torch.tensor(features[indices[:split_idx]], dtype=torch.float32),
        torch.tensor(targets[indices[:split_idx]], dtype=torch.float32),
        torch.tensor(features[indices[split_idx:]], dtype=torch.float32),
        torch.tensor(targets[indices[split_idx:]], dtype=torch.float32),
        scalers
    )

def save_model(model, filename):
    """Save model weights to file."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(model, filename):
    """Load model weights from file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} does not exist")
    model.load_state_dict(torch.load(filename))
    return model


class EarlyStopping:
    def __init__(self, patience=100, verbose=True):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None
        self.verbose = verbose

    def step(self, loss, model):
        if loss < self.best_loss:
            if self.verbose:
                print(f"Validation loss improved from {self.best_loss:.6f} to {loss:.6f}")
            self.best_loss = loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose and self.counter > 0:
                print(f"Validation loss did not improve from {self.best_loss:.6f}, counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience
    


def train_pinn(pinn, x_train, u_train, x_val, u_val, epochs, lr, patience=50, save_path=None, device='mps', verbose=True, print_interval=50, snapshot_num=None, hidden_dim=64, num_layers=4):
    """Train PINN model with early stopping and save epoch losses."""
    pinn = pinn.to(device)
    x_train, u_train, x_val, u_val = x_train.to(device), u_train.to(device), x_val.to(device), u_val.to(device)

    optimizer = optim.Adam(pinn.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience, verbose=verbose)

    train_losses, val_losses = [], []
    epoch_list = []
    
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(epochs):
        def train_step():
            pinn.train()
            optimizer.zero_grad()
            loss = criterion(pinn(x_train), u_train)
            loss.backward()
            optimizer.step()
            return loss.item()

        def val_step():
            pinn.eval()
            with torch.no_grad():
                return criterion(pinn(x_val), u_val).item()

        train_loss, val_loss = train_step(), val_step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_list.append(epoch + 1)

        if verbose and (epoch + 1) % print_interval == 0:
            elapsed = datetime.now() - start_time
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {elapsed}, ETA: {eta}')

        if early_stopping.step(val_loss, pinn):
            print(f"Early stopping at epoch {epoch+1}")
            break

    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training completed in {training_time}, finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    pinn.load_state_dict(early_stopping.best_model)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate a filename with all the requested information
    filename = f"pinn_model_{timestamp}_sn{snapshot_num}_e{epochs}_hd{hidden_dim}_nl{num_layers}.pt"
    
    # Handle save_path properly
    if save_path:
        # If save_path is a directory, use it as is
        if os.path.isdir(save_path) or not save_path.endswith('.pt'):
            os.makedirs(save_path, exist_ok=True)
            final_save_path = os.path.join(save_path, filename)
        else:
            # If save_path looks like a file path, get its directory
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                final_save_path = os.path.join(save_dir, filename)
            else:
                # If no directory was specified, save in the current directory
                final_save_path = filename
    else:
        # No save_path provided, save in current directory
        final_save_path = filename
    
    save_model(pinn, final_save_path)
    print(f"Model saved to {final_save_path}")
    
    # Plot loss curves
    plot_loss_curves(epoch_list, train_losses, val_losses, snapshot_num=snapshot_num, hidden_dim=hidden_dim, num_layers=num_layers)
    
    # Save epoch losses
    loss_file = save_epoch_losses(epoch_list, train_losses, val_losses, snapshot_num=snapshot_num, hidden_dim=hidden_dim, num_layers=num_layers)

    return pinn, final_save_path  # Return model and save path

def generate_filename(base_name, timestamp, snapshot_num, epochs, hidden_dim, num_layers, ext):
    """Generate a filename with timestamp and model details."""
    return f"{base_name}_{timestamp}_sn{snapshot_num}_e{epochs}_hd{hidden_dim}_nl{num_layers}.{ext}"


def save_epoch_losses(epochs, train_losses, val_losses, snapshot_num, hidden_dim, num_layers, output_dir='loss_logs'):
    """
    Save training and validation losses for each epoch to a CSV file.
    
    Args:
        epochs: List of epoch numbers
        train_losses: List of training losses
        val_losses: List of validation losses
        snapshot_num: Number of snapshots used
        hidden_dim: Hidden dimension of the neural network
        num_layers: Number of hidden layers
        output_dir: Directory to save the loss log file
    """
    try:
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_epoch = epochs[-1] if isinstance(epochs, list) else epochs
        filename = f"epoch_losses_{timestamp}_sn{snapshot_num}_e{last_epoch}_hd{hidden_dim}_nl{num_layers}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Epoch': epochs,
            'Train_Loss': train_losses,
            'Validation_Loss': val_losses
        })
        
        # 保存到CSV
        df.to_csv(filepath, index=False)
        print(f"Epoch losses saved to {filepath}")
        
        return filepath
    
    except Exception as e:
        print(f"Error saving epoch losses: {e}")
        raise


def plot_loss_curves(epochs, train_losses, val_losses, snapshot_num=None, hidden_dim=64, num_layers=4):
    """Plot training and validation loss curves with log scale visualization only."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    last_epoch = epochs[-1]  # 获取当前训练的 epochs 数
    
    plt.figure(figsize=(16, 4))
    
    # Log scale plot
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (log scale)', fontsize=14)
    plt.title(f'Training and Validation Loss (Log Scale) - {last_epoch} Epochs', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    
    # 保存文件，包含时间戳和 epochs
    last_epoch = epochs[-1] if isinstance(epochs, list) else epochs
    filename = generate_filename("loss_curve", timestamp, snapshot_num, last_epoch, hidden_dim, num_layers, ext='pdf')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curve saved as {filename}")


def predict_displacement(pinn, G, K, coords, scalers, device='mps'):
    """Predict displacements for given G, K values and coordinates."""
    pinn = pinn.to(device)
    
    try:
        x_scaler = scalers['x-coordinate [mm]']
        y_scaler = scalers['y-coordinate [mm]']
        G_scaler = scalers['G']
        K_scaler = scalers['K']
        ux_scaler = scalers['x-displacement [mm]']
        uy_scaler = scalers['y-displacement [mm]']
        
        # Scale coordinates
        x_scaled = x_scaler.transform(coords[:, 0].reshape(-1, 1))
        y_scaled = y_scaler.transform(coords[:, 1].reshape(-1, 1))
        coords_scaled = np.hstack((x_scaled, y_scaled))
        
        # Scale G and K values
        G_scaled = G_scaler.transform(np.array([[G]]))
        K_scaled = K_scaler.transform(np.array([[K]]))
        
        # Repeat G, K values for each coordinate point
        GK = np.tile(np.hstack((G_scaled, K_scaled)), (coords.shape[0], 1))
        
        # Combine coordinates and material properties
        x_pred = torch.tensor(np.hstack((coords_scaled, GK)), dtype=torch.float32).to(device)
        
        # Make predictions
        pinn.eval()
        with torch.no_grad():
            u_pred_scaled = pinn(x_pred).cpu().numpy()
        
        # Inverse transform to get actual displacement values
        ux_pred = ux_scaler.inverse_transform(u_pred_scaled[:, 0].reshape(-1, 1))
        uy_pred = uy_scaler.inverse_transform(u_pred_scaled[:, 1].reshape(-1, 1))
        u_pred = np.hstack((ux_pred, uy_pred))
        
        return u_pred
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


def plot_displacement_comparison(predictions_file, hole_center=(40, 10), hole_radius=4, epochs=None, hidden_dim=64, num_layers=4, snapshot_num=None, G=None, K=None):
    """
    Plot advanced displacement field visualization with separate plots for X and Y components
    for easier comparison with FEM results.
    """
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file {predictions_file} does not exist")
    
    try:
        # Load prediction data
        df = pd.read_csv(predictions_file)
        x = df['x-coordinate [mm]']
        y = df['y-coordinate [mm]']
        ux = df['x-displacement [mm]']
        uy = df['y-displacement [mm]']

        # Create circular hole mask
        center_x, center_y = hole_center
        radius = hole_radius
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = center_x + radius * np.cos(theta)
        circle_y = center_y + radius * np.sin(theta)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_epoch = epochs[-1] if isinstance(epochs, list) else epochs  # 取最后一个 epoch

        
        # Plot X-displacement (separate figure)
        fig_x, ax_x = plt.subplots(figsize=(16, 4))
        contour_x = ax_x.tricontourf(x, y, ux, levels=100, cmap='turbo')
        cbar_x = fig_x.colorbar(contour_x, ax=ax_x)
        cbar_x.set_label(label='X-Displacement [mm]', size=14)
        ax_x.set_title(f'Predicted X-Displacement Field\nEpochs: {epochs}_ SN: {snapshot_num}', fontsize=16)
        ax_x.set_xlabel('X Coordinate [mm]', fontsize=14)
        ax_x.set_ylabel('Y Coordinate [mm]', fontsize=14)
        ax_x.grid(True)
        ax_x.fill(circle_x, circle_y, 'w-', lw=2)
        
        plt.tight_layout()
        filename_x = generate_filename("x_displacement", timestamp, snapshot_num, last_epoch, hidden_dim, num_layers, ext='pdf')
        plt.savefig(filename_x, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Y-displacement (separate figure)
        fig_y, ax_y = plt.subplots(figsize=(16, 4))
        contour_y = ax_y.tricontourf(x, y, uy, levels=100, cmap='turbo')
        cbar_y = fig_y.colorbar(contour_y, ax=ax_y)
        cbar_y.set_label(label='Y-Displacement [mm]', size=14)
        ax_y.set_title(f'Predicted Y-Displacement Field\nEpochs: {epochs}_ SN: {snapshot_num}', fontsize=16)
        ax_y.set_xlabel('X Coordinate [mm]', fontsize=14)
        ax_y.set_ylabel('Y Coordinate [mm]', fontsize=14)
        ax_y.grid(True)
        ax_y.fill(circle_x, circle_y, 'w-', lw=2)
        
        plt.tight_layout()
        filename_y = generate_filename("y_displacement", timestamp, snapshot_num, last_epoch, hidden_dim, num_layers, ext='pdf')
        plt.savefig(filename_y, dpi=300, bbox_inches='tight')
        plt.close()

        
        # Plot displacement magnitude (useful for overall deformation analysis)
        fig_mag, ax_mag = plt.subplots(figsize=(16, 4))
        displacement_magnitude = np.sqrt(ux**2 + uy**2)
        contour_mag = ax_mag.tricontourf(x, y, displacement_magnitude, levels=100, cmap='turbo')
        cbar_mag = fig_mag.colorbar(contour_mag, ax=ax_mag)
        cbar_mag.set_label(label='Displacement Magnitude [mm]', size=14)
        ax_mag.set_title(f'Predicted Displacement Magnitude\nEpochs: {epochs}_ SN: {snapshot_num}', fontsize=16)
        ax_mag.set_xlabel('X Coordinate [mm]', fontsize=14)
        ax_mag.set_ylabel('Y Coordinate [mm]', fontsize=14)
        ax_mag.grid(True)
        ax_mag.fill(circle_x, circle_y, 'w-', lw=2)
        
        plt.tight_layout()
        filename_mag = generate_filename("displacement_magnitude", timestamp, snapshot_num, last_epoch, hidden_dim, num_layers, ext='pdf')
        plt.savefig(filename_mag, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved displacement plots: \n{filename_x}\n{filename_y}\n{filename_mag}")
        
    except Exception as e:
        print(f"Error plotting predictions: {e}")
        raise


def save_predictions(coords, u_pred, G, K, filename='predictions.csv'):
    """Save displacement predictions to CSV file."""
    try:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        df = pd.DataFrame({
            'x-coordinate [mm]': coords[:, 0],
            'y-coordinate [mm]': coords[:, 1],
            'x-displacement [mm]': u_pred[:, 0],
            'y-displacement [mm]': u_pred[:, 1]
        })
        df.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")
        return df
    except Exception as e:
        print(f"Error saving predictions: {e}")
        raise


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PINN Model for Displacement Prediction')

    parser.add_argument('--no_mps', action='store_true', help='Disable MPS acceleration even if available')
    parser.add_argument('--exp_data_file', type=str, default=None, help='Path to experimental displacement data CSV file')
    
    # 数据相关参数
    parser.add_argument('--snapshots_dir', type=str, default='snapshots', help='Directory containing snapshot files')
    parser.add_argument('--snapshots_num', type=int, default=50, help='Number of snapshots to use (random selection)')
    parser.add_argument('--combined_csv', type=str, default='combined_data.csv', help='Path to save/load combined data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--force_combine', action='store_true', help='Force recombination of snapshots even if combined file exists')
    
    # 模型相关参数
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of the neural network')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of hidden layers')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience')
    parser.add_argument('--print_interval', type=int, default=50, help='Interval for printing training progress')
    parser.add_argument('--model_file', type=str, default='pinn_model.pt', help='Path to save/load model')
    parser.add_argument('--no_train', action='store_true', help='Skip training and load model instead')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage even if available')
    
    # 预测相关参数
    parser.add_argument('--hole_center_x', type=float, default=40, help='X-coordinate of hole center')
    parser.add_argument('--hole_center_y', type=float, default=10, help='Y-coordinate of hole center')
    parser.add_argument('--hole_radius', type=float, default=4, help='Radius of the hole')
    
    # 解析参数
    args, unknown = parser.parse_known_args()
    
    # 处理格式为 param:value 的额外参数
    for arg in unknown:
        if ':' in arg:
            key, value = arg.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # 尝试转换为数值类型
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # 保持为字符串
                pass
                
            # 设置到args
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")
    
    return args

def main():
    args = parse_args()

    print("\n--- Configuration ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-------------------\n")

    # 确定设备
    if torch.backends.mps.is_available() and not args.no_gpu and not args.no_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available() and not args.no_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    try:
        # 合并快照并训练模型
        if not os.path.exists(args.combined_csv) or args.force_combine:
            print(f"{'Forcing recombination' if args.force_combine else 'Combined data file not found'}. Combining snapshots...")
            df, combined_file = combine_snapshots_to_csv(
                args.snapshots_dir, 
                args.combined_csv, 
                args.snapshots_num,  # 修改为 args.snapshots_num
                last_epoch=args.epochs, 
                hidden_dim=args.hidden_dim, 
                num_layers=args.num_layers
            )
            # 获取实际使用的快照数量
            snapshot_num = args.snapshots_num if args.snapshots_num is not None else len([f for f in os.listdir(args.snapshots_dir) if re.match(r'snapshot_G[\d\.eE+-]+_K[\d\.eE+-]+\.csv', f)])
        else:
            print(f"Loading combined data from {args.combined_csv}...")
            df = pd.read_csv(args.combined_csv)
            combined_file = args.combined_csv
            # 从文件名中提取 snapshot_num（如果存在）
            match = re.search(r'_sn(\d+)_', combined_file)
            snapshot_num = int(match.group(1)) if match else args.snapshots_num  # 修改为 args.snapshots_num

        x_train, u_train, x_val, u_val, scalers = load_and_preprocess_data(df, test_size=args.test_size)
        print(f"Train data: x_train shape: {x_train.shape}, u_train shape: {u_train.shape}")
        print(f"Validation data: x_val shape: {x_val.shape}, u_val shape: {u_val.shape}")

        # 创建模型
        pinn = PINN(args.hidden_dim, args.num_layers)
        print(f"Model created with hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")

        # 训练或加载模型
        if not args.no_train:
            print(f"Training PINN model for {args.epochs} epochs...")
            pinn, model_filename = train_pinn(
                pinn, x_train, u_train, x_val, u_val,
                args.epochs, args.lr, args.patience,
                save_path=args.model_file,
                device=device,
                print_interval=args.print_interval,
                snapshot_num=snapshot_num,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers
            )
        else:
            if os.path.exists(args.model_file):
                print(f"Loading model from {args.model_file}...")
                pinn = load_model(pinn, args.model_file)
                pinn = pinn.to(device)
            else:
                raise FileNotFoundError(f"Model file {args.model_file} not found. Set --no_train=False to train a new model.")

        # --- 添加预测和绘图逻辑 ---
        x_coords = np.linspace(0, 80, 100)
        y_coords = np.linspace(0, 20, 25)
        X, Y = np.meshgrid(x_coords, y_coords)
        coords = np.vstack((X.flatten(), Y.flatten())).T

        G = 7.35e10
        K = 1.28e11
        if 'G' in df.columns and 'K' in df.columns:
            G = df['G'].iloc[0]
            K = df['K'].iloc[0]

        u_pred = predict_displacement(pinn, G, K, coords, scalers, device=device)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_file = f"predictions_{timestamp}.csv"
        pred_df = save_predictions(coords, u_pred, G, K, filename=pred_file)

        plot_displacement_comparison(
            pred_file,
            hole_center=(args.hole_center_x, args.hole_center_y),
            hole_radius=args.hole_radius,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            snapshot_num=snapshot_num,
            G=G,
            K=K
        )

        print("\nProcess completed successfully!")

    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()