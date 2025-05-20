import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('global_housing_market_extended.csv')

# 数据预处理
# 移除非数值列
data_numeric = data.select_dtypes(include=[np.number])

# 提取特征和目标
features = ['Rent Index', 'Affordability Ratio', 'Mortgage Rate (%)', 'Inflation Rate (%)',
            'GDP Growth (%)', 'Population Growth (%)', 'Urbanization Rate (%)', 'Construction Index']
target = 'House Price Index'

X = data_numeric[features]
y = data_numeric[target]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 相关性分析
correlation_matrix = data_numeric.corr()
house_price_corr = correlation_matrix[target].sort_values(ascending=False)

# 可视化相关性
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 特征重要性评估
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
feature_importances = model.feature_importances_

# 可视化特征重要性
feature_names = np.array(features)
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(12, 4))  # 增加图表宽度
plt.bar(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.xticks(range(len(sorted_idx)), feature_names[sorted_idx], rotation=45, fontsize=10)  # 减少旋转角度并调整字体大小
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Feature Importances', fontsize=14)
plt.tight_layout()
plt.show()

# 打印相关性和特征重要性结果
print("Correlation with House Price Index:")
print(house_price_corr)
print("\nFeature Importances:")
for feature, importance in zip(feature_names[sorted_idx], feature_importances[sorted_idx]):
    print(f"{feature}: {importance:.4f}")
