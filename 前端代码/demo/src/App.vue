<template>
  <div class="container">
    <h1 class="title1">数据挖掘终期汇报</h1>
    
    <!-- 第一部分：房价预测分析 -->
    <section class="section">
      <h2 class="title2">1、全球房价预测分析</h2>
      <div class="selector-row">
        <label for="region-select">选择地区：</label>
        <select id="region-select" v-model="selectedRegion">
          <option v-for="region in regions" :key="region.value" :value="region.value">
            {{ region.label }}
          </option>
        </select>
      </div>
      <div class="main-image-card" v-if="currentImage">
        <div class="loading-spinner" v-if="isLoading">
          <div class="spinner"></div>
          <!-- <div class="loading-text">加载中...</div> -->
        </div>
        <img 
          :src="currentImage.url" 
          :alt="currentImage.name" 
          class="main-image" 
          @click="toggleFullscreen('prediction')"
          @load="handleImageLoad"
          @error="handleImageError"
          :class="{ 'image-hidden': isLoading }"
        >
        <div class="main-image-info">{{ currentImage.name }}</div>
      </div>
    </section>

    <!-- 第二部分：相关性分析 -->
    <section class="section">
      <h2 class="title2">2、相关性分析</h2>
      <div class="main-image-card">
        <img 
          :src="correlationImage" 
          alt="相关性分析" 
          class="main-image" 
          @click="toggleFullscreen('correlation')"
        >
        <div class="analysis-dropdown">
          <div class="dropdown-header" @click="toggleAnalysis('correlation')">
            点击查看相关性分析结果
            <i class="arrow" :class="{ 'arrow-down': !showCorrelationAnalysis, 'arrow-up': showCorrelationAnalysis }"></i>
          </div>
          <div class="dropdown-content" v-show="showCorrelationAnalysis">
            <p>1. 相关性分析的核心目标是揭示住房市场数据中各个特征变量与房价指数之间的线性关系。通过计算皮尔逊相关系数，程序能够评估每个特征与房价指数之间的相关性强度和方向。</p>
          </div>
        </div>
      </div>
    </section>

    <!-- 第三部分：特征重要性评估 -->
    <section class="section">
      <h2 class="title2">3、特征重要性评估</h2>
      <div class="main-image-card">
        <img 
          :src="importanceImage" 
          alt="特征重要性评估" 
          class="main-image" 
          @click="toggleFullscreen('importance')"
        >
        <div class="analysis-dropdown">
          <div class="dropdown-header" @click="toggleAnalysis('importance')">
            点击查看特征重要性评估结果
            <i class="arrow" :class="{ 'arrow-down': !showImportanceAnalysis, 'arrow-up': showImportanceAnalysis }"></i>
          </div>
          <div class="dropdown-content" v-show="showImportanceAnalysis">
            <p>1. 特征重要性评估采用了随机森林回归模型（Random Forest Regressor）</p>
            <p>2. 如果一个特征能够有效地减少决策树中的不纯度（例如，减少均方误差），则说明该特征对于模型的预测能力有较大的影响，反之则为较小的贡献</p>
            <p>3. 建设指数和GDP增长率是最为重要的两个特征，其重要性值较高，表明它们对模型预测的贡献最大</p>
            <p>4. 抵押贷款利率和房屋可负担性比率的重要性稍低，但仍具有一定影响力</p>
            <p>5. 租赁指数、城市化率、人口增长率和通货膨胀率的重要性依次递减，表明它们对模型预测的影响相对较弱</p>
          </div>
        </div>
      </div>
    </section>

    <!-- 第四部分：聚类分析 -->
    <section class="section">
      <h2 class="title2">4、聚类分析</h2>
      
      <!-- 第一阶段：肘部法则 -->
      <div class="main-image-card">
        <h3 class="cluster-subtitle">4.1 肘部法则确定最佳聚类数</h3>
        <img 
          :src="clusterImages[1]" 
          alt="肘部法则分析" 
          class="main-image" 
          @click="toggleFullscreen('cluster1')"
        >
        <div class="analysis-dropdown">
          <div class="dropdown-header" @click="toggleAnalysis('cluster1')">
            点击查看肘部法则分析结果
            <i class="arrow" :class="{ 'arrow-down': !showCluster1Analysis, 'arrow-up': showCluster1Analysis }"></i>
          </div>
          <div class="dropdown-content" v-show="showCluster1Analysis">
            <p>通过肘部法则（Elbow Method）分析不同K值下的组内平方和（WSS），我们可以观察到：</p>
            <p>1. 当K值从2增加到4时，WSS显著下降</p>
            <p>2. 在K=4之后，WSS的下降趋势明显放缓</p>
            <p>3. 这表明K=4是一个较为合适的聚类数量，能够在聚类效果和计算复杂度之间取得良好的平衡</p>
          </div>
        </div>
      </div>

      <!-- 分析结果展示 -->
      <h3 class="cluster-subtitle">4.2 聚类分析结果</h3>
      <div class="cluster-results">
        <!-- 左侧分析结果 -->
        <div class="result-card">
          <h4 class="result-title">基于经济指标的聚类</h4>
          <img 
            :src="clusterImages[2]" 
            alt="聚类分析结果1" 
            class="result-image" 
            @click="toggleFullscreen('cluster2')"
          >
          <div class="analysis-dropdown">
            <div class="dropdown-header" @click="toggleAnalysis('cluster2')">
              点击查看分析结果
              <i class="arrow" :class="{ 'arrow-down': !showCluster2Analysis, 'arrow-up': showCluster2Analysis }"></i>
            </div>
            <div class="dropdown-content" v-show="showCluster2Analysis">
              <p>  将8维特征压缩为2个主成分，展示各国在二维空间中的分布及其聚类归属，并详细标注主成分的含义和解释率。 </p>
              <p>  4个分组（聚类）依据是每个国家在2023年下列8个关键指标上的数值相似性： 房价指数（House Price Index） 租金指数（Rent Index） 可负担比率（Affordability Ratio） 抵押贷款利率（Mortgage Rate ） 通货膨胀率（Inflation Rate ） GDP增长率（GDP Growth ） 城市化率（Urbanization Rate ） 建筑指数（Construction Index）</p>
            </div>
          </div>
        </div>

        <!-- 右侧分析结果 -->
        <div class="result-card">
          <h4 class="result-title">基于房价趋势的聚类</h4>
          <img 
            :src="clusterImages[3]" 
            alt="聚类分析结果2" 
            class="result-image" 
            @click="toggleFullscreen('cluster3')"
          >
          <div class="analysis-dropdown">
            <div class="dropdown-header" @click="toggleAnalysis('cluster3')">
              点击查看分析结果
              <i class="arrow" :class="{ 'arrow-down': !showCluster3Analysis, 'arrow-up': showCluster3Analysis }"></i>
            </div>
            <div class="dropdown-content" v-show="showCluster3Analysis">
              <p>雷达图：展示每个聚类在8个特征上的相对表现，并对每个特征做了详细中文说明，便于对比不同聚类的特征优势。</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 全屏预览模态框 -->
    <div v-if="fullscreenImage" class="fullscreen-modal" @click="closeFullscreen">
      <div class="fullscreen-content">
        <img :src="fullscreenImage" :alt="fullscreenAlt">
      </div>
    </div>
  </div>
</template>

<script>
// 导入所有图片
import Australia from '@/assets/Results/Australia_house_price_prediction.png'
import Brazil from '@/assets/Results/Brazil_house_price_prediction.png'
import Canada from '@/assets/Results/Canada_house_price_prediction.png'
import China from '@/assets/Results/China_house_price_prediction.png'
import France from '@/assets/Results/France_house_price_prediction.png'
import Germany from '@/assets/Results/Germany_house_price_prediction.png'
import India from '@/assets/Results/India_house_price_prediction.png'
import Italy from '@/assets/Results/Italy_house_price_prediction.png'
import Japan from '@/assets/Results/Japan_house_price_prediction.png'
import Mexico from '@/assets/Results/Mexico_house_price_prediction.png'
import Netherlands from '@/assets/Results/Netherlands_house_price_prediction.png'
import Russia from '@/assets/Results/Russia_house_price_prediction.png'
import SouthAfrica from '@/assets/Results/South Africa_house_price_prediction.png'
import SouthKorea from '@/assets/Results/South Korea_house_price_prediction.png'
import Spain from '@/assets/Results/Spain_house_price_prediction.png'
import Sweden from '@/assets/Results/Sweden_house_price_prediction.png'
import Switzerland from '@/assets/Results/Switzerland_house_price_prediction.png'
import UAE from '@/assets/Results/UAE_house_price_prediction.png'
import UK from '@/assets/Results/UK_house_price_prediction.png'
import USA from '@/assets/Results/USA_house_price_prediction.png'

import Correlation from '@/assets/Results/相关性分析.png'
import Importance from '@/assets/Results/特征重要性评估.png'
import Cluster1 from '@/assets/Results/聚类1.png'
import Cluster2 from '@/assets/Results/聚类2.png'
import Cluster3 from '@/assets/Results/聚类3.png'

export default {
  name: 'App',
  data() {
    return {
      regions: [
        { label: '澳大利亚', value: 'Australia' },
        { label: '巴西', value: 'Brazil' },
        { label: '加拿大', value: 'Canada' },
        { label: '中国', value: 'China' },
        { label: '法国', value: 'France' },
        { label: '德国', value: 'Germany' },
        { label: '印度', value: 'India' },
        { label: '意大利', value: 'Italy' },
        { label: '日本', value: 'Japan' },
        { label: '墨西哥', value: 'Mexico' },
        { label: '荷兰', value: 'Netherlands' },
        { label: '俄罗斯', value: 'Russia' },
        { label: '南非', value: 'SouthAfrica' },
        { label: '韩国', value: 'SouthKorea' },
        { label: '西班牙', value: 'Spain' },
        { label: '瑞典', value: 'Sweden' },
        { label: '瑞士', value: 'Switzerland' },
        { label: '阿联酋', value: 'UAE' },
        { label: '英国', value: 'UK' },
        { label: '美国', value: 'USA' }
      ],
      images: {
        Australia: { name: '澳大利亚房价预测', url: Australia },
        Brazil: { name: '巴西房价预测', url: Brazil },
        Canada: { name: '加拿大房价预测', url: Canada },
        China: { name: '中国房价预测', url: China },
        France: { name: '法国房价预测', url: France },
        Germany: { name: '德国房价预测', url: Germany },
        India: { name: '印度房价预测', url: India },
        Italy: { name: '意大利房价预测', url: Italy },
        Japan: { name: '日本房价预测', url: Japan },
        Mexico: { name: '墨西哥房价预测', url: Mexico },
        Netherlands: { name: '荷兰房价预测', url: Netherlands },
        Russia: { name: '俄罗斯房价预测', url: Russia },
        SouthAfrica: { name: '南非房价预测', url: SouthAfrica },
        SouthKorea: { name: '韩国房价预测', url: SouthKorea },
        Spain: { name: '西班牙房价预测', url: Spain },
        Sweden: { name: '瑞典房价预测', url: Sweden },
        Switzerland: { name: '瑞士房价预测', url: Switzerland },
        UAE: { name: '阿联酋房价预测', url: UAE },
        UK: { name: '英国房价预测', url: UK },
        USA: { name: '美国房价预测', url: USA }
      },
      selectedRegion: 'China',
      isFullscreen: false,
      isLoading: true,
      loadingMinTime: 1000, // 最小加载时间（毫秒）
      loadingStartTime: 0, // 记录加载开始时间
      correlationImage: Correlation,
      importanceImage: Importance,
      fullscreenImage: null,
      fullscreenAlt: '',
      showCorrelationAnalysis: false,
      showImportanceAnalysis: false,
      showCluster1Analysis: false,
      showCluster2Analysis: false,
      showCluster3Analysis: false,
      clusterImages: {
        1: Cluster1, // 肘部法则图
        2: Cluster2, // 第一种聚类结果
        3: Cluster3  // 第二种聚类结果
      }
    }
  },
  computed: {
    currentImage() {
      return this.images[this.selectedRegion]
    }
  },
  methods: {
    toggleFullscreen(type) {
      if (type === 'prediction') {
        this.fullscreenImage = this.currentImage.url;
        this.fullscreenAlt = this.currentImage.name;
      } else if (type === 'correlation') {
        this.fullscreenImage = this.correlationImage;
        this.fullscreenAlt = '相关性分析';
      } else if (type === 'importance') {
        this.fullscreenImage = this.importanceImage;
        this.fullscreenAlt = '特征重要性评估';
      } else if (type === 'cluster1') {
        this.fullscreenImage = this.clusterImages[1];
        this.fullscreenAlt = '肘部法则分析';
      } else if (type === 'cluster2') {
        this.fullscreenImage = this.clusterImages[2];
        this.fullscreenAlt = '聚类分析结果1';
      } else if (type === 'cluster3') {
        this.fullscreenImage = this.clusterImages[3];
        this.fullscreenAlt = '聚类分析结果2';
      }
    },
    closeFullscreen() {
      this.fullscreenImage = null;
      this.fullscreenAlt = '';
    },
    startLoading() {
      this.isLoading = true;
      this.loadingStartTime = Date.now();
    },
    handleImageLoad() {
      const currentTime = Date.now();
      const elapsedTime = currentTime - this.loadingStartTime;
      
      if (elapsedTime < this.loadingMinTime) {
        // 如果加载时间小于最小时间，延迟关闭加载动画
        setTimeout(() => {
          this.isLoading = false;
        }, this.loadingMinTime - elapsedTime);
      } else {
        // 如果已经超过最小时间，直接关闭加载动画
        this.isLoading = false;
      }
    },
    handleImageError() {
      this.handleImageLoad(); // 复用相同的时间控制逻辑
    },
    toggleAnalysis(type) {
      if (type === 'correlation') {
        this.showCorrelationAnalysis = !this.showCorrelationAnalysis;
      } else if (type === 'importance') {
        this.showImportanceAnalysis = !this.showImportanceAnalysis;
      } else if (type === 'cluster1') {
        this.showCluster1Analysis = !this.showCluster1Analysis;
      } else if (type === 'cluster2') {
        this.showCluster2Analysis = !this.showCluster2Analysis;
      } else if (type === 'cluster3') {
        this.showCluster3Analysis = !this.showCluster3Analysis;
      }
    },
  },
  watch: {
    selectedRegion() {
      this.startLoading();
    }
  },
  mounted() {
    this.startLoading();
    
    // 添加滚动动画
    const sections = document.querySelectorAll('.section');
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.animationDelay = '0.2s';
          entry.target.style.animationPlayState = 'running';
        }
      });
    }, { threshold: 0.1 });

    sections.forEach(section => {
      section.style.animationPlayState = 'paused';
      observer.observe(section);
    });
  }
}
</script>

<style>
.container {
  max-width: 1200px; /* 增加容器最大宽度 */
  margin: 0 auto;
  padding: 30px 20px;
  background: linear-gradient(45deg, #f6f8fb, #f5f5f5, #e8f0fe);
  background-size: 400% 400%;
  animation: gradientBG 15s ease infinite;
  min-height: 100vh;
}

@keyframes gradientBG {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.title1 {
  text-align: center;
  color: #2c3e50;
  margin-bottom: 50px;
  font-size: 2.5em;
  font-weight: bold;
  background: linear-gradient(120deg, #2c3e50, #3498db, #2c3e50);
  background-size: 200% auto;
  color: transparent;
  -webkit-background-clip: text;
  background-clip: text;
  animation: shine 3s linear infinite;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

@keyframes shine {
  to { background-position: 200% center; }
}

.title2 {
  text-align: center;
  color: #2c3e50;
  margin-bottom: 30px;
  font-size: 2em;
  font-weight: bold;
}

.selector-row {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 30px;
  gap: 10px;
}

.selector-row label {
  font-size: 1.2em;
  color: #333;
  font-weight: 500;
}

.selector-row select {
  font-size: 1.2em;
  padding: 8px 20px;
  border-radius: 8px;
  border: 1px solid #bbb;
  outline: none;
  transition: all 0.3s;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(5px);
  border: 2px solid transparent;
  background-image: linear-gradient(white, white), 
                    linear-gradient(120deg, #3498db, #2c3e50);
  background-origin: border-box;
  background-clip: padding-box, border-box;
  cursor: pointer;
}

.selector-row select:hover {
  border-color: #409eff;
}

.selector-row select:focus {
  border-color: #409eff;
  box-shadow: 0 0 0 2px rgba(64,158,255,0.2);
}

.main-image-card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  padding: 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 0 auto;
  width: 95%;
  position: relative;
  min-height: 600px; /* 确保加载时容器有高度 */
  transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
}

.main-image-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}

.main-image {
  width: 100%;
  min-height: 600px;
  max-height: 800px;
  object-fit: contain;
  border-radius: 8px;
  background: #fafafa;
  margin-bottom: 18px;
  cursor: pointer;
  transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  opacity: 1;
}

.main-image:hover {
  transform: scale(1.03);
  box-shadow: 0 10px 20px rgba(0,0,0,0.1);
}

.main-image-info {
  font-size: 1.3em;
  color: #2c3e50;
  text-align: center;
  font-weight: 500;
  margin-top: 10px;
}

/* 全屏预览模态框样式 */
.fullscreen-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  cursor: pointer;
  backdrop-filter: blur(5px);
  animation: modalFade 0.3s ease-in-out;
}

@keyframes modalFade {
  from { opacity: 0; }
  to { opacity: 1; }
}

.fullscreen-content {
  width: 95%;
  height: 95%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.fullscreen-content img {
  max-width: 95%;
  max-height: 95%;
  object-fit: contain;
}

/* 加载动画样式 */
.loading-spinner {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 3px solid transparent;
  border-top: 3px solid #3498db;
  border-right: 3px solid #2c3e50;
  border-bottom: 3px solid #3498db;
  animation: spinGradient 1s linear infinite;
}

.loading-text {
  color: #409eff;
  font-size: 1.2em;
  font-weight: 500;
}

@keyframes spinGradient {
  0% { transform: rotate(0deg); border-top-color: #3498db; }
  50% { border-top-color: #2c3e50; }
  100% { transform: rotate(360deg); border-top-color: #3498db; }
}

.image-hidden {
  opacity: 0;
}

@media (max-width: 768px) {
  .container {
    padding: 15px 10px;
  }

  .main-image {
    min-height: 400px;
  }

  .title {
    font-size: 2em;
  }

  .selector-row select {
    font-size: 1em;
    padding: 6px 12px;
  }
}

.section {
  margin-bottom: 60px;
  opacity: 0;
  transform: translateY(20px);
  animation: fadeInUp 0.8s ease forwards;
}

@keyframes fadeInUp {
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.section:last-child {
  margin-bottom: 30px;
}

.analysis-dropdown {
  width: 100%;
  margin-top: 20px;
}

.dropdown-header {
  background-color: #f5f7fa;
  padding: 15px 20px;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 1.1em;
  color: #409eff;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.dropdown-header:hover {
  background-color: #ecf5ff;
}

.dropdown-header::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(120deg, transparent, rgba(255,255,255,0.2), transparent);
  transform: translateX(-100%);
}

.dropdown-header:hover::after {
  transform: translateX(100%);
  transition: transform 0.6s ease;
}

.dropdown-content {
  background-color: #fff;
  padding: 20px;
  border-radius: 0 0 8px 8px;
  margin-top: 2px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  transform-origin: top;
  animation: dropDown 0.3s ease-out;
}

.dropdown-content p {
  margin: 10px 0;
  line-height: 1.6;
  color: #2c3e50;
  font-size: 1.05em;
}

.arrow {
  border: solid #409eff;
  border-width: 0 2px 2px 0;
  display: inline-block;
  padding: 3px;
  margin-left: 10px;
  transition: transform 0.3s ease;
}

.arrow-down {
  transform: rotate(45deg);
}

.arrow-up {
  transform: rotate(-135deg);
}

/* 确保下拉内容的平滑过渡 */
.dropdown-content {
  transition: all 0.3s ease-in-out;
  overflow: hidden;
}

/* 添加新的样式 */
.cluster-subtitle {
  text-align: center;
  color: #2c3e50;
  margin: 40px 0 20px;
  font-size: 1.5em;
  font-weight: bold;
  position: relative;
  display: inline-block;
  padding: 0 10px;
}

.cluster-subtitle::after {
  content: '';
  position: absolute;
  bottom: -5px;
  left: 0;
  width: 100%;
  height: 2px;
  background: linear-gradient(90deg, transparent, #3498db, transparent);
}

.cluster-results {
  display: flex;
  gap: 30px;
  margin-top: 20px;
  perspective: 1000px;
}

.result-card {
  flex: 1;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  padding: 24px;
  transform-style: preserve-3d;
  transition: transform 0.5s ease;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
}

.result-card:hover {
  transform: rotateY(5deg);
}

.result-title {
  text-align: center;
  color: #2c3e50;
  margin-bottom: 20px;
  font-size: 1.2em;
  font-weight: 500;
}

.result-image {
  width: 100%;
  height: auto;
  min-height: 300px;
  object-fit: contain;
  border-radius: 8px;
  cursor: pointer;
  transition: transform 0.3s ease;
  margin-bottom: 20px;
}

.result-image:hover {
  transform: scale(1.02);
}

/* 响应式调整 */
@media (max-width: 1024px) {
  .cluster-results {
    flex-direction: column;
  }

  .result-card {
    margin-bottom: 30px;
  }
}
</style>
