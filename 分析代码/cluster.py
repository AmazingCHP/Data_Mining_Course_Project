# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ======================
# 1. 数据加载与预处理
# ======================
# 加载实际数据
df = pd.read_csv('global_housing_market_extended.csv')

# 选择2023年的数据进行分析
df_2023 = df[df['Year'] == 2023]

# 选择用于聚类的特征
features = ['House Price Index', 'Rent Index', 'Affordability Ratio', 
           'Mortgage Rate (%)', 'Inflation Rate (%)', 'GDP Growth (%)', 
           'Population Growth (%)', 'Urbanization Rate (%)', 'Construction Index']
X = df_2023[features]

# 标准化数据
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

# ======================
# 2. 确定最佳聚类数（肘部法则）
# ======================
sse = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

# 计算SSE变化率（差分）
delta_sse = np.diff(sse)

# 自动推荐K值（最大变化率的下一个K）
if len(delta_sse) > 0:
    recommend_k = int(np.argmin(delta_sse) + 2)  # +2是因为diff后索引偏移
else:
    recommend_k = 2

plt.figure(figsize=(12, 7))
plt.plot(K, sse, marker='o', linestyle='--', color='#1f77b4', label='SSE')
plt.xlabel('聚类数量 (K)\n选择拐点处的K值作为最佳聚类数', fontsize=12)
plt.ylabel('聚类内误差平方和 (SSE)\n值越小表示聚类效果越好', fontsize=12)
plt.title('使用肘部法则确定最佳聚类数\n在曲线拐点处为最佳K值', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(K)
plt.tight_layout()

# 绘制SSE变化率辅助线
plt.twinx()
plt.plot(K[1:], -delta_sse, marker='x', linestyle='-', color='#ff7f0e', label='SSE下降幅度')
plt.ylabel('SSE下降幅度（相邻K的差值）', fontsize=12)

# 标记推荐K值
plt.axvline(recommend_k, color='red', linestyle=':', linewidth=2, label=f'推荐K={recommend_k}')
plt.legend(loc='best')
plt.show()

# ======================
# 3. 执行K-means聚类
# ======================
k = 4  # 根据肘部图选择
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df_2023['Cluster'] = clusters

# 打印聚类中心（原始尺度）
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(
    cluster_centers,
    columns=features,
    index=[f'Cluster {i}' for i in range(k)]
)

# 打印每个聚类的详细信息
print("\n=== 聚类结果详细分析 ===")
for cluster in range(k):
    print(f"\n【聚类 {cluster}】")
    countries = df_2023[df_2023['Cluster'] == cluster]['Country'].tolist()
    print(f"包含的国家：{', '.join(countries)}")
    
    # 显示该组的原始均值（非标准化）
    cluster_means = df_2023[df_2023['Cluster'] == cluster][features].mean()
    print("\n关键指标均值：")
    for feature in features:
        print(f"{feature}: {cluster_means[feature]:.2f}")

# ======================
# 4. 可视化聚类结果
# ======================
# 4.1 二维散点图（PCA降维）
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# 计算主成分的方差解释率
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x=pca_features[:, 0], 
    y=pca_features[:, 1], 
    hue=df_2023['Cluster'], 
    palette='viridis', 
    s=150,
    edgecolor='k',
    alpha=0.8
)
plt.title('2023年全球房地产市场聚类分析\n使用PCA降维将8个特征压缩为2个主成分', fontsize=14)
plt.xlabel(f'主成分1 (解释{explained_variance_ratio[0]:.1%}的数据方差)\n主要反映房价、租金和经济指标的综合表现', fontsize=12)
plt.ylabel(f'主成分2 (解释{explained_variance_ratio[1]:.1%}的数据方差)\n主要反映城市化和建设指标的综合表现', fontsize=12)

# 标记国家名称
for i, (x, y, country) in enumerate(zip(pca_features[:, 0], pca_features[:, 1], df_2023['Country'])):
    plt.text(x, y, country, fontsize=9, ha='center', va='center')

plt.legend(title='聚类分组', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4.2 改进的雷达图
def plot_radar_chart(cluster_means, feature_names):
    # 使用MinMaxScaler进行0-1标准化，保持相对关系
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(cluster_means)
    radar_data = pd.DataFrame(normalized_data, 
                            columns=feature_names, 
                            index=cluster_means.index)
    
    # 为特征名称添加更详细的说明
    feature_descriptions = {
        'House Price Index': f'房价指数\n(组均值范围:{cluster_means["House Price Index"].min():.0f}-{cluster_means["House Price Index"].max():.0f})',
        'Rent Index': f'租金指数\n(组均值范围:{cluster_means["Rent Index"].min():.0f}-{cluster_means["Rent Index"].max():.0f})',
        'Affordability Ratio': f'可负担比率\n(组均值范围:{cluster_means["Affordability Ratio"].min():.1f}-{cluster_means["Affordability Ratio"].max():.1f})',
        'Mortgage Rate (%)': f'抵押贷款利率\n(组均值范围:{cluster_means["Mortgage Rate (%)"].min():.1f}-{cluster_means["Mortgage Rate (%)"].max():.1f}%)',
        'Inflation Rate (%)': f'通货膨胀率\n(组均值范围:{cluster_means["Inflation Rate (%)"].min():.1f}-{cluster_means["Inflation Rate (%)"].max():.1f}%)',
        'GDP Growth (%)': f'GDP增长率\n(组均值范围:{cluster_means["GDP Growth (%)"].min():.1f}-{cluster_means["GDP Growth (%)"].max():.1f}%)',
        'Population Growth (%)': f'人口增长率\n(组均值范围:{cluster_means["Population Growth (%)"].min():.1f}-{cluster_means["Population Growth (%)"].max():.1f}%)',
        'Urbanization Rate (%)': f'城市化率\n(组均值范围:{cluster_means["Urbanization Rate (%)"].min():.1f}-{cluster_means["Urbanization Rate (%)"].max():.1f}%)',
        'Construction Index': f'建筑指数\n(组均值范围:{cluster_means["Construction Index"].min():.0f}-{cluster_means["Construction Index"].max():.0f})'
    }
    
    feature_labels = [feature_descriptions[f] for f in feature_names]
    
    num_vars = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'polar': True})
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (cluster, row) in enumerate(radar_data.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        
        # 获取该组的国家列表
        countries = df_2023[df_2023['Cluster'] == idx]['Country'].tolist()
        label = f'聚类 {idx}\n({", ".join(countries)})'
        
        ax.plot(angles, values, color=colors[idx], linewidth=2, label=label)
        ax.fill(angles, values, color=colors[idx], alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=9)
    plt.title('各聚类特征对比\n显示原始数值范围，标准化至0-1区间以便对比', fontsize=14, pad=20)
    ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), title='聚类分组及包含的国家')
    plt.tight_layout()
    plt.show()

# 计算每个簇的均值
cluster_means = df_2023.groupby('Cluster')[features].mean()
plot_radar_chart(cluster_means, features)

# 保存详细的聚类结果
output_df = df_2023[['Country', 'Cluster'] + features].copy()
output_df.sort_values(['Cluster', 'Country'], inplace=True)
output_df.to_csv('clustering_results_2023_detailed.csv', index=False)