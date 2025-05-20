import pandas as pd

# 读取数据
data = pd.read_csv('global_housing_market_extended.csv')

# 排除非数值列（Country和Year）
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# 计算每个数值特征的最小值和最大值
ranges = pd.DataFrame({
    # 'Feature': numeric_columns,
    'Min': data[numeric_columns].min(),
    'Max': data[numeric_columns].max()
})

print(ranges)
