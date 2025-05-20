import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 确保保存图片的目录存在
output_dir = "house_price_plots"
os.makedirs(output_dir, exist_ok=True)

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据加载和预处理
class HousingDataset(Dataset):
    def __init__(self, filepath, sequence_length=3):
        self.data = pd.read_csv(filepath)
        self.sequence_length = sequence_length
        self.features = ['House Price Index', 'Rent Index', 'Affordability Ratio',
                         'Mortgage Rate (%)', 'Inflation Rate (%)', 'GDP Growth (%)',
                         'Population Growth (%)', 'Urbanization Rate (%)', 'Construction Index']
        self.target = 'House Price Index'

        # 按国家和年份排序
        self.data = self.data.sort_values(['Country', 'Year'])

        # 为每个国家单独标准化数据
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = []
        self.countries = []
        self.original_values = []

        for country in self.data['Country'].unique():
            country_data = self.data[self.data['Country'] == country][self.features].values
            scaled_country_data = self.scaler.fit_transform(country_data)
            self.scaled_data.append(scaled_country_data)
            self.countries.extend([country] * len(country_data))
            self.original_values.append(country_data)

        self.scaled_data = np.concatenate(self.scaled_data)
        self.original_values = np.concatenate(self.original_values)
        self.countries = np.array(self.countries)

        # 创建序列
        self.X, self.y, self.country_seq = self._create_sequences()

    def _create_sequences(self):
        X, y = [], []
        country_seq = []
        unique_countries = np.unique(self.countries)

        for country in unique_countries:
            country_indices = np.where(self.countries == country)[0]
            country_data = self.scaled_data[country_indices]

            for i in range(len(country_data) - self.sequence_length):
                X.append(country_data[i:i + self.sequence_length])
                y.append(country_data[i + self.sequence_length, 0])  # 预测House Price Index
                country_seq.append(country)

        return np.array(X), np.array(y), np.array(country_seq)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])


# 2. LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out


# 3. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=10):
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # 计算训练损失
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    return model


# 4. 评估函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

    return total_loss / len(data_loader.dataset)


# 加载CSV文件
def load_data_from_csv(csv_file):
    # 假设CSV包含列：Country, Year, House_Price
    df = pd.read_csv(csv_file)
    return df


def test_and_visualize(model, dataset, device, scaler, countries, csv_file):
    model.eval()
    all_preds = []
    all_targets = []
    country_list = []

    # 存储预测的2025年房价
    predictions_2025 = {}

    # 从CSV加载历史数据
    df = load_data_from_csv(csv_file)

    with torch.no_grad():
        for i in range(len(dataset)):
            inputs, target = dataset[i]
            inputs = inputs.unsqueeze(0).to(device)
            output = model(inputs)

            # 反标准化
            dummy_array = np.zeros((1, len(dataset.dataset.features)))  # 访问原始数据集的 features 属性

            dummy_array[:, 0] = output.cpu().numpy()
            pred = scaler.inverse_transform(dummy_array)[0, 0]

            dummy_array[:, 0] = target.numpy()
            true_val = scaler.inverse_transform(dummy_array)[0, 0]

            all_preds.append(pred)
            all_targets.append(true_val)
            country_list.append(countries[i])

            # 存储每个国家的2025年预测房价
            country = countries[i]
            if country not in predictions_2025:
                predictions_2025[country] = []
            predictions_2025[country].append(pred)

    print(predictions_2025)

    # 计算指标
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # 按国家可视化结果
    unique_countries = np.unique(country_list)

    for country in unique_countries:
        country_indices = np.where(np.array(country_list) == country)[0]
        plt.figure(figsize=(12, 6))

        # 获取2015-2024年的历史数据
        country_data = df[(df['Country'] == country) & (df['Year'] >= 2015) & (df['Year'] <= 2024)]

        if len(country_data) == 10:
            # 提取历史数据的房价信息
            country_data_values = country_data['House Price Index'].values

            # 获取预测的2025年数据
            country_predictions_2025 = predictions_2025[country][0]
            country_data_15_25 = np.concatenate([country_data_values, [country_predictions_2025]])

            # 绘制15-25年的房价折线图
            plt.plot(np.arange(2015, 2025), country_data_values, label='Actual (2015-2024)', color='blue')
            plt.plot(np.arange(2025, 2026), country_predictions_2025, label='Predicted (2025)', color='red', marker='o')
            plt.plot(np.arange(2015, 2026), country_data_15_25, label='Prediction (2015-2025)', linestyle='--',
                     color='green')

            plt.title(f'House Price Index Prediction - {country}')
            plt.xlabel('Year')
            plt.ylabel('House Price Index')
            plt.legend()
            filename = os.path.join(output_dir, f"{country}_house_price_prediction.png")
            plt.savefig(filename)
            plt.close()

        else:
            print(f"Warning: Not enough data for country {country} to plot correctly.")

    return all_preds, all_targets


# 主函数
def main():
    # 参数设置
    SEQUENCE_LENGTH = 3
    BATCH_SIZE = 32
    EPOCHS = 200
    PATIENCE = 15
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据集
    dataset = HousingDataset('global_housing_market_extended.csv', SEQUENCE_LENGTH)

    # 2. 划分训练集、验证集和测试集
    # 首先按国家分层划分训练集和测试集
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.country_seq
    )

    # 再从训练集中划分验证集
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # 创建子数据集
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化模型
    input_size = len(dataset.features)
    model = LSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    # 4. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 训练模型
    print("Starting training...")
    model = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        epochs=EPOCHS, patience=PATIENCE
    )

    # 6. 在测试集上评估
    print("\nEvaluating on test set...")
    test_countries = np.array(dataset.country_seq)[test_idx]
    test_preds, test_targets = test_and_visualize(
        model, test_dataset, device,
        dataset.scaler, test_countries, "global_housing_market_extended.csv"
    )

    # 7. 保存模型（可选）
    torch.save(model.state_dict(), 'house_price_lstm_pytorch.pth')


if __name__ == "__main__":
    main()
