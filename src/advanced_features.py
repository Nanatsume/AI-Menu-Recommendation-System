"""
Advanced Features Extension for AI Menu Recommendation System
ฟีเจอร์ขั้นสูงเพิ่มเติมสำหรับระบบแนะนำเมนู
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """คลาสสำหรับการแบ่งกลุ่มลูกค้า"""
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.segment_labels = None
        
    def create_customer_features(self, df_orders, df_customers, df_menu):
        """สร้าง features สำหรับลูกค้า"""
        try:
            # สถิติการสั่งอาหาร - ตรวจสอบ columns ที่มีจริง
            agg_dict = {}
            
            # ใช้ columns ที่มีจริง
            if 'rating' in df_orders.columns:
                agg_dict['rating'] = ['count', 'mean', 'std']
            else:
                # ใช้ amount แทนเพื่อนับจำนวนครั้ง
                agg_dict['amount'] = ['count', 'mean', 'std']
                
            if 'quantity' in df_orders.columns:
                agg_dict['quantity'] = ['sum', 'mean']
            else:
                # ใช้ค่าเริ่มต้น
                agg_dict['amount'] = agg_dict.get('amount', []) + ['sum']
                
            if 'order_date' in df_orders.columns:
                agg_dict['order_date'] = ['min', 'max']
            
            order_stats = df_orders.groupby('customer_id').agg(agg_dict).round(2)
            
            # ตั้งชื่อ columns ตามข้อมูลที่มี
            if 'rating' in df_orders.columns:
                order_stats.columns = [
                    'order_frequency', 'avg_rating', 'rating_std',
                    'total_quantity', 'avg_quantity', 'first_order', 'last_order'
                ]
            else:
                if 'quantity' in df_orders.columns:
                    order_stats.columns = [
                        'order_frequency', 'avg_amount', 'amount_std',
                        'total_quantity', 'avg_quantity', 'first_order', 'last_order'
                    ]
                else:
                    order_stats.columns = [
                        'order_frequency', 'avg_amount', 'amount_std', 'total_amount',
                        'first_order', 'last_order'
                    ]
            
            # คำนวณ recency (วันที่สั่งล่าสุด) ถ้ามี order_date
            if 'order_date' in df_orders.columns:
                try:
                    order_stats['recency'] = (pd.Timestamp.now() - pd.to_datetime(order_stats['last_order'])).dt.days
                except:
                    order_stats['recency'] = 0  # ค่าเริ่มต้น
            else:
                order_stats['recency'] = 0
            
            # ความหลากหลายของเมนู - ทำให้ง่ายขึ้น
            diversity_stats = pd.DataFrame(index=df_orders['customer_id'].unique())
            
            # นับความหลากหลายของเมนู
            if 'menu_name' in df_orders.columns:
                diversity_stats['menu_diversity'] = df_orders.groupby('customer_id')['menu_name'].nunique()
            elif 'menu_id' in df_orders.columns:
                diversity_stats['menu_diversity'] = df_orders.groupby('customer_id')['menu_id'].nunique()
            else:
                diversity_stats['menu_diversity'] = 1
            
            # ราคาเฉลี่ย
            if 'amount' in df_orders.columns:
                diversity_stats['avg_price'] = df_orders.groupby('customer_id')['amount'].mean()
            else:
                diversity_stats['avg_price'] = 100
            
            # ความหลากหลายหมวดหมู่ - ไม่บังคับ
            if 'category' in df_orders.columns:
                diversity_stats['category_diversity'] = df_orders.groupby('customer_id')['category'].nunique()
            else:
                # ไม่จำเป็นต้องมี category - ใช้ค่าเริ่มต้น
                diversity_stats['category_diversity'] = 1
            
            # รวมข้อมูล
            customer_features = df_customers.set_index('customer_id').join([
                order_stats.fillna(0),
                diversity_stats.fillna(0)
            ], how='left').fillna(0)
            
            return customer_features
            
        except Exception as e:
            print(f"Error in create_customer_features: {e}")
            # สร้าง features พื้นฐาน
            basic_features = df_customers.set_index('customer_id')
            basic_features['order_frequency'] = 1
            basic_features['avg_rating'] = 4.0
            basic_features['total_quantity'] = 1
            basic_features['recency'] = 0
            basic_features['category_diversity'] = 1
            basic_features['menu_diversity'] = 1
            basic_features['avg_price'] = 100
            return basic_features
    
    def fit_segments(self, customer_features):
        """แบ่งกลุ่มลูกค้า"""
        try:
            # เลือก features สำคัญที่มีอยู่จริง
            available_cols = customer_features.columns.tolist()
            
            # Base features
            feature_cols = ['age', 'avg_budget', 'order_frequency']
            
            # เพิ่ม rating หรือ amount
            if 'avg_rating' in available_cols:
                feature_cols.append('avg_rating')
            elif 'avg_amount' in available_cols:
                feature_cols.append('avg_amount')
            
            # เพิ่ม quantity
            if 'total_quantity' in available_cols:
                feature_cols.append('total_quantity')
            elif 'total_amount' in available_cols:
                feature_cols.append('total_amount')
            
            # เพิ่ม features อื่นๆ ที่มี
            optional_features = ['recency', 'category_diversity', 'avg_price', 'menu_diversity']
            for col in optional_features:
                if col in available_cols:
                    feature_cols.append(col)
            
            print(f"Using features for clustering: {feature_cols}")
            
            # ใช้เฉพาะ features ที่มีจริง
            X = customer_features[feature_cols].fillna(0)
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # K-means clustering
            cluster_labels = self.kmeans.fit_predict(X_scaled)
            
            # กำหนด segment labels
            self.segment_labels = self._assign_segment_names(customer_features, cluster_labels)
            
            return cluster_labels, self.segment_labels
            
        except Exception as e:
            print(f"Error in fit_segments: {e}")
            # สร้าง random clusters ถ้าเกิด error
            n_customers = len(customer_features)
            cluster_labels = np.random.randint(0, self.n_clusters, n_customers)
            self.segment_labels = {i: f"กลุ่ม {i+1}" for i in range(self.n_clusters)}
            return cluster_labels, self.segment_labels
    
    def _assign_segment_names(self, features, cluster_labels):
        """กำหนดชื่อ segment ตามลักษณะแต่ละ cluster"""
        segment_stats = features.copy()
        segment_stats['cluster'] = cluster_labels
        
        cluster_summary = segment_stats.groupby('cluster').agg({
            'order_frequency': 'mean',
            'avg_rating': 'mean',
            'avg_budget': 'mean',
            'recency': 'mean'
        })
        
        print(f"Cluster summary:\n{cluster_summary}")
        
        # คำนวณ percentiles สำหรับการจัดกลุ่ม
        freq_75 = features['order_frequency'].quantile(0.75)
        freq_50 = features['order_frequency'].quantile(0.50)
        rating_75 = features['avg_rating'].quantile(0.75)
        budget_75 = features['avg_budget'].quantile(0.75)
        budget_50 = features['avg_budget'].quantile(0.50)
        
        segment_names = {}
        for cluster in cluster_summary.index:
            stats = cluster_summary.loc[cluster]
            
            # กำหนดชื่อที่ชัดเจนตาม cluster พร้อมหมายเลข
            if stats['order_frequency'] >= freq_75 and stats['avg_budget'] >= budget_75:
                segment_names[cluster] = f'🏆 VIP Champions (กลุ่ม {cluster + 1})'
            elif stats['order_frequency'] >= freq_75:
                segment_names[cluster] = f'🔄 Frequent Buyers (กลุ่ม {cluster + 1})'
            elif stats['avg_budget'] >= budget_75:
                segment_names[cluster] = f'💰 High Spenders (กลุ่ม {cluster + 1})'
            elif stats['order_frequency'] >= freq_50 and stats['avg_rating'] >= rating_75:
                segment_names[cluster] = f'💎 Loyal Customers (กลุ่ม {cluster + 1})'
            elif stats['order_frequency'] >= freq_50:
                segment_names[cluster] = f'👥 Regular Customers (กลุ่ม {cluster + 1})'
            else:
                segment_names[cluster] = f'🌱 New Customers (กลุ่ม {cluster + 1})'
        
        print(f"Assigned segment names: {segment_names}")
        return segment_names
    
    def plot_segments(self, customer_features, cluster_labels):
        """สร้างกราฟแสดงการแบ่งกลุ่ม"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Order Frequency vs Avg Rating
        scatter = axes[0, 0].scatter(
            customer_features['order_frequency'],
            customer_features['avg_rating'],
            c=cluster_labels,
            cmap='viridis',
            alpha=0.7
        )
        axes[0, 0].set_xlabel('Order Frequency')
        axes[0, 0].set_ylabel('Average Rating')
        axes[0, 0].set_title('🎯 Customer Segments: Frequency vs Rating')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Budget vs Age
        scatter = axes[0, 1].scatter(
            customer_features['age'],
            customer_features['avg_budget'],
            c=cluster_labels,
            cmap='viridis',
            alpha=0.7
        )
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Average Budget')
        axes[0, 1].set_title('💰 Customer Segments: Age vs Budget')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. Recency vs Total Quantity
        scatter = axes[1, 0].scatter(
            customer_features['recency'],
            customer_features['total_quantity'],
            c=cluster_labels,
            cmap='viridis',
            alpha=0.7
        )
        axes[1, 0].set_xlabel('Recency (days)')
        axes[1, 0].set_ylabel('Total Quantity')
        axes[1, 0].set_title('⏰ Customer Segments: Recency vs Quantity')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # 4. Category Diversity Distribution
        for cluster in np.unique(cluster_labels):
            mask = cluster_labels == cluster
            axes[1, 1].hist(
                customer_features[mask]['category_diversity'],
                alpha=0.7,
                label=f'Cluster {cluster}'
            )
        axes[1, 1].set_xlabel('Category Diversity')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('🌈 Category Diversity by Segment')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()


class ContentBasedRecommender:
    """Content-based recommendation engine"""
    
    def __init__(self):
        self.item_features = None
        self.similarity_matrix = None
        
    def create_item_features(self, df_menu):
        """สร้าง features สำหรับเมนู"""
        # One-hot encoding สำหรับ category
        category_dummies = pd.get_dummies(df_menu['category'], prefix='category')
        
        # Normalize price และ popularity
        features = df_menu[['menu_id', 'price', 'popularity']].copy()
        features['price_normalized'] = (features['price'] - features['price'].min()) / (features['price'].max() - features['price'].min())
        features['popularity_normalized'] = features['popularity'] / features['popularity'].max()
        
        # รวม features
        self.item_features = pd.concat([
            features[['menu_id', 'price_normalized', 'popularity_normalized']],
            category_dummies
        ], axis=1).set_index('menu_id')
        
        return self.item_features
    
    def compute_similarity(self):
        """คำนวณ similarity matrix"""
        if self.item_features is None:
            raise ValueError("Item features not created yet!")
        
        self.similarity_matrix = cosine_similarity(self.item_features.values)
        return self.similarity_matrix
    
    def recommend_similar_items(self, item_id, top_k=10):
        """แนะนำเมนูที่คล้ายกัน"""
        if self.similarity_matrix is None:
            self.compute_similarity()
        
        # หา index ของเมนู
        item_indices = list(self.item_features.index)
        if item_id not in item_indices:
            return []
        
        item_idx = item_indices.index(item_id)
        
        # หา similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[item_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # เลือก top-k (ไม่รวมตัวเอง)
        similar_items = []
        for i, score in sim_scores[1:top_k+1]:
            similar_item_id = item_indices[i]
            similar_items.append((similar_item_id, score))
        
        return similar_items


class TrendAnalyzer:
    """วิเคราะห์เทรนด์และพฤติกรรม"""
    
    def __init__(self):
        pass
    
    def analyze_temporal_trends(self, df_orders, df_menu):
        """วิเคราะห์เทรนด์ตามเวลา"""
        df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
        df_orders['month'] = df_orders['order_date'].dt.month
        df_orders['hour'] = df_orders['order_date'].dt.hour
        df_orders['day_of_week'] = df_orders['order_date'].dt.day_name()
        
        trends = {}
        
        # เทรนด์รายเดือน
        monthly_trend = df_orders.groupby('month').agg({
            'rating': ['count', 'mean'],
            'quantity': 'sum'
        }).round(2)
        trends['monthly'] = monthly_trend
        
        # เทรนด์รายชั่วโมง
        hourly_trend = df_orders.groupby('hour').agg({
            'rating': ['count', 'mean'],
            'quantity': 'sum'
        }).round(2)
        trends['hourly'] = hourly_trend
        
        # เทรนด์รายวัน
        daily_trend = df_orders.groupby('day_of_week').agg({
            'rating': ['count', 'mean'],
            'quantity': 'sum'
        }).round(2)
        trends['daily'] = daily_trend
        
        return trends
    
    def plot_trends(self, trends):
        """สร้างกราฟแสดงเทรนด์"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly trend
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_orders = trends['monthly'][('rating', 'count')]
        axes[0, 0].plot(range(1, 13), monthly_orders, marker='o', linewidth=2)
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Number of Orders')
        axes[0, 0].set_title('📅 Monthly Order Trends')
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].set_xticklabels(months[:len(monthly_orders)])
        axes[0, 0].grid(alpha=0.3)
        
        # Hourly trend
        hourly_orders = trends['hourly'][('rating', 'count')]
        axes[0, 1].bar(hourly_orders.index, hourly_orders.values, alpha=0.7)
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Number of Orders')
        axes[0, 1].set_title('⏰ Hourly Order Distribution')
        axes[0, 1].grid(alpha=0.3)
        
        # Daily trend
        daily_orders = trends['daily'][('rating', 'count')]
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        reordered_daily = daily_orders.reindex([day for day in days_order if day in daily_orders.index])
        axes[1, 0].bar(range(len(reordered_daily)), reordered_daily.values, alpha=0.7)
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Number of Orders')
        axes[1, 0].set_title('📊 Daily Order Distribution')
        axes[1, 0].set_xticks(range(len(reordered_daily)))
        axes[1, 0].set_xticklabels([day[:3] for day in reordered_daily.index], rotation=45)
        axes[1, 0].grid(alpha=0.3)
        
        # Rating trends by hour
        hourly_ratings = trends['hourly'][('rating', 'mean')]
        axes[1, 1].plot(hourly_ratings.index, hourly_ratings.values, 
                       marker='s', color='orange', linewidth=2)
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Average Rating')
        axes[1, 1].set_title('⭐ Rating Trends by Hour')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def find_peak_hours(self, trends):
        """หาช่วงเวลาที่มีออเดอร์มากที่สุด"""
        hourly_orders = trends['hourly'][('rating', 'count')]
        peak_hours = hourly_orders.nlargest(3)
        
        print("🔥 Peak Hours (Top 3):")
        for hour, count in peak_hours.items():
            time_period = "เช้า" if 6 <= hour < 12 else "บ่าย" if 12 <= hour < 18 else "เย็น" if 18 <= hour < 24 else "กลางคืน"
            print(f"   {hour:02d}:00 ({time_period}) - {count} orders")
        
        return peak_hours.index.tolist()


class ABTestingFramework:
    """Framework สำหรับ A/B Testing"""
    
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(self, name, models, test_ratio=0.5):
        """สร้าง A/B test experiment"""
        self.experiments[name] = {
            'models': models,
            'test_ratio': test_ratio,
            'results': {}
        }
        
        print(f"🧪 สร้าง A/B Test: {name}")
        print(f"   Models: {list(models.keys())}")
        print(f"   Test Ratio: {test_ratio}")
    
    def run_experiment(self, experiment_name, user_item_matrix, df_menu, df_orders, n_users=100):
        """รัน A/B test"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found!")
        
        exp = self.experiments[experiment_name]
        models = exp['models']
        
        # สุ่มเลือกผู้ใช้สำหรับทดสอบ
        test_users = np.random.choice(user_item_matrix.index, n_users, replace=False)
        
        results = {}
        
        print(f"\n🚀 รัน A/B Test: {experiment_name}")
        print(f"   Testing with {n_users} users...")
        
        for model_name, model in models.items():
            print(f"\n📊 Testing {model_name}...")
            
            # ประเมินโมเดล
            from src.evaluation_fixed import AdvancedRecommendationEvaluator
            evaluator = AdvancedRecommendationEvaluator(model)
            
            # ใช้ subset ของ user-item matrix
            test_matrix = user_item_matrix.loc[test_users]
            test_results = evaluator.evaluate_comprehensive(test_matrix, df_menu, df_orders, test_ratio=0.2)
            
            results[model_name] = test_results
        
        self.experiments[experiment_name]['results'] = results
        
        return results
    
    def compare_results(self, experiment_name):
        """เปรียบเทียบผลลัพธ์ A/B test"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found!")
        
        results = self.experiments[experiment_name]['results']
        
        if not results:
            print("❌ No results found. Run experiment first!")
            return
        
        print(f"\n📊 A/B Test Results: {experiment_name}")
        print("="*60)
        
        # เปรียบเทียบ key metrics
        key_metrics = ['precision@10', 'recall@10', 'ndcg@10', 'mrr', 'diversity']
        
        comparison_df = pd.DataFrame(results).T[key_metrics]
        
        print("\n🏆 Model Comparison:")
        print(comparison_df.round(4))
        
        # หา winner
        overall_scores = comparison_df.mean(axis=1)
        winner = overall_scores.idxmax()
        
        print(f"\n🥇 Winner: {winner} (Score: {overall_scores[winner]:.4f})")
        
        # สร้างกราฟเปรียบเทียบ
        fig, ax = plt.subplots(figsize=(12, 6))
        comparison_df.plot(kind='bar', ax=ax)
        ax.set_title(f'📊 A/B Test Results: {experiment_name}')
        ax.set_ylabel('Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return winner, comparison_df
