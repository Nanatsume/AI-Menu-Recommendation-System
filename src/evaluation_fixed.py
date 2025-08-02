"""
Enhanced Evaluation Module for AI Menu Recommendation System
ประเมินผลการทำงานของระบบแนะนำเมนูแบบครบถ้วน
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# การตั้งค่า font สำหรับการแสดงผลภาษาไทย
plt.rcParams['font.family'] = ['DejaVu Sans', 'Tahoma']

class AdvancedRecommendationEvaluator:
    """คลาสสำหรับประเมินผลระบบแนะนำแบบครบถ้วน"""
    
    def __init__(self, recommendation_model=None):
        """
        Initialize the evaluator
        
        Parameters:
        recommendation_model: โมเดลที่จะทำการประเมิน
        """
        self.model = recommendation_model
        self.evaluation_results = {}
        
    def precision_at_k(self, recommended_items, relevant_items, k_values=[5, 10, 20]):
        """
        คำนวณ Precision@K สำหรับ k หลายค่า
        
        Precision@K = จำนวนเมนูที่แนะนำถูกต้องใน K อันดับแรก / K
        """
        results = {}
        for k in k_values:
            if k == 0 or len(recommended_items) == 0:
                results[f'precision@{k}'] = 0.0
                continue
                
            recommended_k = recommended_items[:k]
            relevant_recommended = set(recommended_k) & set(relevant_items)
            results[f'precision@{k}'] = len(relevant_recommended) / min(k, len(recommended_items))
        
        return results
    
    def recall_at_k(self, recommended_items, relevant_items, k_values=[5, 10, 20]):
        """
        คำนวณ Recall@K สำหรับ k หลายค่า
        
        Recall@K = จำนวนเมนูที่แนะนำถูกต้องใน K อันดับแรก / จำนวนเมนูที่เกี่ยวข้องทั้งหมด
        """
        results = {}
        for k in k_values:
            if len(relevant_items) == 0:
                results[f'recall@{k}'] = 0.0
                continue
                
            recommended_k = recommended_items[:k]
            relevant_recommended = set(recommended_k) & set(relevant_items)
            results[f'recall@{k}'] = len(relevant_recommended) / len(relevant_items)
        
        return results
    
    def f1_at_k(self, recommended_items, relevant_items, k_values=[5, 10, 20]):
        """คำนวณ F1-Score@K ซึ่งเป็น harmonic mean ของ Precision และ Recall"""
        precision_results = self.precision_at_k(recommended_items, relevant_items, k_values)
        recall_results = self.recall_at_k(recommended_items, relevant_items, k_values)
        
        f1_results = {}
        for k in k_values:
            p = precision_results[f'precision@{k}']
            r = recall_results[f'recall@{k}']
            
            if p + r == 0:
                f1_results[f'f1@{k}'] = 0.0
            else:
                f1_results[f'f1@{k}'] = 2 * (p * r) / (p + r)
        
        return f1_results
    
    def hit_rate_at_k(self, recommended_items, relevant_items, k_values=[5, 10, 20]):
        """
        คำนวณ Hit Rate@K 
        
        Hit Rate@K = 1 ถ้ามีเมนูที่เกี่ยวข้องอย่างน้อย 1 รายการใน K อันดับแรก, 0 ถ้าไม่มี
        """
        results = {}
        for k in k_values:
            recommended_k = recommended_items[:k]
            hit = 1.0 if len(set(recommended_k) & set(relevant_items)) > 0 else 0.0
            results[f'hit_rate@{k}'] = hit
        
        return results
    
    def mean_reciprocal_rank(self, recommended_items, relevant_items):
        """
        คำนวณ Mean Reciprocal Rank (MRR)
        
        MRR = 1 / อันดับของเมนูที่เกี่ยวข้องตัวแรกในรายการแนะนำ
        """
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    def ndcg_at_k(self, recommended_items, relevant_items, k_values=[5, 10, 20]):
        """
        คำนวณ Normalized Discounted Cumulative Gain (NDCG@K)
        
        NDCG พิจารณาทั้งความถูกต้องและตำแหน่งของเมนูที่แนะนำ
        """
        results = {}
        
        for k in k_values:
            # DCG calculation
            dcg = 0.0
            for i, item in enumerate(recommended_items[:k]):
                if item in relevant_items:
                    dcg += 1.0 / np.log2(i + 2)  # +2 เพราะ log2(1) = 0
            
            # IDCG calculation (ideal DCG)
            idcg = 0.0
            for i in range(min(k, len(relevant_items))):
                idcg += 1.0 / np.log2(i + 2)
            
            # NDCG
            if idcg == 0:
                results[f'ndcg@{k}'] = 0.0
            else:
                results[f'ndcg@{k}'] = dcg / idcg
        
        return results
    
    def catalog_coverage(self, all_recommendations, total_items):
        """
        คำนวณ Catalog Coverage - ร้อยละของเมนูที่ถูกแนะนำ
        
        Coverage = จำนวนเมนูเฉพาะที่ถูกแนะนำ / จำนวนเมนูทั้งหมด
        """
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update(recs)
        
        return len(unique_recommended) / total_items
    
    def diversity_score(self, recommendations, item_features):
        """
        คำนวณ Diversity Score - ความหลากหลายของเมนูที่แนะนำ
        
        วัดจากความแตกต่างของหมวดหมู่เมนูในรายการแนะนำ
        """
        if len(recommendations) < 2:
            return 0.0
        
        total_pairs = 0
        diverse_pairs = 0
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                total_pairs += 1
                
                # เปรียบเทียบหมวดหมู่เมนู
                item1_category = item_features.get(recommendations[i], {}).get('category', '')
                item2_category = item_features.get(recommendations[j], {}).get('category', '')
                
                if item1_category != item2_category:
                    diverse_pairs += 1
        
        return diverse_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def novelty_score(self, recommendations, item_popularity):
        """
        คำนวณ Novelty Score - ความแปลกใหม่ของเมนูที่แนะนำ
        
        เมนูที่ไม่ค่อยมีคนสั่งจะมี novelty สูง
        """
        if len(recommendations) == 0:
            return 0.0
        
        novelty = 0.0
        for item in recommendations:
            # novelty = -log2(popularity)
            popularity = item_popularity.get(item, 0.1)  # default ต่ำถ้าไม่มีข้อมูล
            novelty += -np.log2(max(popularity, 0.001))  # ป้องกัน log(0)
        
        return novelty / len(recommendations)
    
    def evaluate_comprehensive(self, user_item_matrix, df_menu, df_orders, test_ratio=0.2):
        """
        ประเมินผลแบบครบถ้วน
        
        Parameters:
        user_item_matrix: DataFrame ของ user-item interaction
        df_menu: DataFrame ของข้อมูลเมนู
        df_orders: DataFrame ของข้อมูลการสั่งซื้อ
        test_ratio: สัดส่วนข้อมูลทดสอบ
        """
        print("🔍 เริ่มการประเมินผลแบบครบถ้วน...")
        
        if self.model is None:
            print("❌ ไม่มีโมเดลสำหรับประเมิน")
            return None
        
        # แบ่งข้อมูลเป็น train/test
        test_users = user_item_matrix.sample(frac=test_ratio).index
        
        all_metrics = {
            'precision': defaultdict(list),
            'recall': defaultdict(list), 
            'f1': defaultdict(list),
            'hit_rate': defaultdict(list),
            'ndcg': defaultdict(list),
            'mrr': [],
            'diversity': [],
            'novelty': []
        }
        
        # คำนวณ item popularity สำหรับ novelty
        item_popularity = df_orders.groupby('menu_id').size() / len(df_orders)
        
        # สร้าง item features สำหรับ diversity
        item_features = df_menu.set_index('menu_id')[['category']].to_dict('index')
        
        all_recommendations = []
        
        print(f"📊 ประเมินผลสำหรับ {len(test_users)} ผู้ใช้...")
        
        for user_id in test_users:
            try:
                # หา user index สำหรับโมเดล
                user_idx = list(user_item_matrix.index).index(user_id)
                
                # รับคำแนะนำจากโมเดล
                recommendations = self.model.predict_for_user(user_idx, top_k=20)
                
                # แปลงเป็น list ของ item IDs
                recommended_items = [item[0] if isinstance(item, tuple) else item 
                                   for item in recommendations]
                
                # หาเมนูที่เคยสั่ง (relevant items)
                relevant_items = user_item_matrix.loc[user_id]
                relevant_items = relevant_items[relevant_items > 0].index.tolist()
                
                # คำนวณ metrics ต่างๆ
                precision_results = self.precision_at_k(recommended_items, relevant_items)
                recall_results = self.recall_at_k(recommended_items, relevant_items)
                f1_results = self.f1_at_k(recommended_items, relevant_items)
                hit_rate_results = self.hit_rate_at_k(recommended_items, relevant_items)
                ndcg_results = self.ndcg_at_k(recommended_items, relevant_items)
                
                # เก็บผลลัพธ์
                for metric, values in precision_results.items():
                    all_metrics['precision'][metric].append(values)
                for metric, values in recall_results.items():
                    all_metrics['recall'][metric].append(values)
                for metric, values in f1_results.items():
                    all_metrics['f1'][metric].append(values)
                for metric, values in hit_rate_results.items():
                    all_metrics['hit_rate'][metric].append(values)
                for metric, values in ndcg_results.items():
                    all_metrics['ndcg'][metric].append(values)
                
                all_metrics['mrr'].append(self.mean_reciprocal_rank(recommended_items, relevant_items))
                all_metrics['diversity'].append(self.diversity_score(recommended_items[:10], item_features))
                all_metrics['novelty'].append(self.novelty_score(recommended_items[:10], item_popularity))
                
                all_recommendations.append(recommended_items)
                
            except Exception as e:
                print(f"⚠️ Error evaluating user {user_id}: {e}")
                continue
        
        # คำนวณค่าเฉลี่ยของ metrics ทั้งหมด
        final_results = {
            'precision@5': np.mean(all_metrics['precision']['precision@5']),
            'precision@10': np.mean(all_metrics['precision']['precision@10']),
            'precision@20': np.mean(all_metrics['precision']['precision@20']),
            'recall@5': np.mean(all_metrics['recall']['recall@5']),
            'recall@10': np.mean(all_metrics['recall']['recall@10']),
            'recall@20': np.mean(all_metrics['recall']['recall@20']),
            'f1@5': np.mean(all_metrics['f1']['f1@5']),
            'f1@10': np.mean(all_metrics['f1']['f1@10']),
            'f1@20': np.mean(all_metrics['f1']['f1@20']),
            'hit_rate@5': np.mean(all_metrics['hit_rate']['hit_rate@5']),
            'hit_rate@10': np.mean(all_metrics['hit_rate']['hit_rate@10']),
            'hit_rate@20': np.mean(all_metrics['hit_rate']['hit_rate@20']),
            'ndcg@5': np.mean(all_metrics['ndcg']['ndcg@5']),
            'ndcg@10': np.mean(all_metrics['ndcg']['ndcg@10']),
            'ndcg@20': np.mean(all_metrics['ndcg']['ndcg@20']),
            'mrr': np.mean(all_metrics['mrr']),
            'diversity': np.mean(all_metrics['diversity']),
            'novelty': np.mean(all_metrics['novelty']),
            'catalog_coverage': self.catalog_coverage(all_recommendations, len(df_menu))
        }
        
        self.evaluation_results = final_results
        
        print("✅ การประเมินผลเสร็จสิ้น!")
        return final_results
    
    def print_evaluation_results(self):
        """แสดงผลการประเมิน"""
        if not self.evaluation_results:
            print("❌ ยังไม่มีผลการประเมิน กรุณารัน evaluate_comprehensive ก่อน")
            return
        
        print("\n" + "="*50)
        print("📊 ผลการประเมินระบบแนะนำเมนูอาหาร")
        print("="*50)
        
        print("\n🎯 Accuracy Metrics:")
        print(f"   Precision@5:  {self.evaluation_results['precision@5']:.4f}")
        print(f"   Precision@10: {self.evaluation_results['precision@10']:.4f}")
        print(f"   Precision@20: {self.evaluation_results['precision@20']:.4f}")
        
        print(f"\n   Recall@5:     {self.evaluation_results['recall@5']:.4f}")
        print(f"   Recall@10:    {self.evaluation_results['recall@10']:.4f}")
        print(f"   Recall@20:    {self.evaluation_results['recall@20']:.4f}")
        
        print(f"\n   F1-Score@5:   {self.evaluation_results['f1@5']:.4f}")
        print(f"   F1-Score@10:  {self.evaluation_results['f1@10']:.4f}")
        print(f"   F1-Score@20:  {self.evaluation_results['f1@20']:.4f}")
        
        print("\n🎪 Ranking Metrics:")
        print(f"   Hit Rate@5:   {self.evaluation_results['hit_rate@5']:.4f}")
        print(f"   Hit Rate@10:  {self.evaluation_results['hit_rate@10']:.4f}")
        print(f"   Hit Rate@20:  {self.evaluation_results['hit_rate@20']:.4f}")
        
        print(f"\n   NDCG@5:       {self.evaluation_results['ndcg@5']:.4f}")
        print(f"   NDCG@10:      {self.evaluation_results['ndcg@10']:.4f}")
        print(f"   NDCG@20:      {self.evaluation_results['ndcg@20']:.4f}")
        
        print(f"\n   MRR:          {self.evaluation_results['mrr']:.4f}")
        
        print("\n🌈 Beyond Accuracy:")
        print(f"   Diversity:    {self.evaluation_results['diversity']:.4f}")
        print(f"   Novelty:      {self.evaluation_results['novelty']:.4f}")
        print(f"   Coverage:     {self.evaluation_results['catalog_coverage']:.4f}")
        
        print("\n📝 Interpretation Guide:")
        print("   • Precision: สัดส่วนเมนูที่แนะนำถูกต้อง")
        print("   • Recall: สัดส่วนเมนูที่เกี่ยวข้องที่ถูกแนะนำ")
        print("   • F1-Score: ค่าเฉลี่ยฮาร์โมนิกของ Precision และ Recall")
        print("   • Hit Rate: สัดส่วนผู้ใช้ที่ได้รับคำแนะนำที่ถูกต้อง")
        print("   • NDCG: คำนึงถึงตำแหน่งของคำแนะนำที่ถูกต้อง")
        print("   • MRR: เฉลี่ยของ reciprocal rank ของคำแนะนำแรกที่ถูกต้อง")
        print("   • Diversity: ความหลากหลายของหมวดหมู่เมนู")
        print("   • Novelty: ความแปลกใหม่ของเมนูที่แนะนำ")
        print("   • Coverage: สัดส่วนเมนูทั้งหมดที่ถูกแนะนำ")
    
    def plot_evaluation_metrics(self):
        """สร้างกราฟแสดงผลการประเมิน"""
        if not self.evaluation_results:
            print("❌ ยังไม่มีผลการประเมิน")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Precision, Recall, F1 @K
        k_values = [5, 10, 20]
        precision_values = [self.evaluation_results[f'precision@{k}'] for k in k_values]
        recall_values = [self.evaluation_results[f'recall@{k}'] for k in k_values]
        f1_values = [self.evaluation_results[f'f1@{k}'] for k in k_values]
        
        axes[0, 0].plot(k_values, precision_values, 'bo-', label='Precision@K', linewidth=2)
        axes[0, 0].plot(k_values, recall_values, 'ro-', label='Recall@K', linewidth=2)
        axes[0, 0].plot(k_values, f1_values, 'go-', label='F1-Score@K', linewidth=2)
        axes[0, 0].set_xlabel('K')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('📈 Precision, Recall, F1-Score @K', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Hit Rate และ NDCG @K
        hit_rate_values = [self.evaluation_results[f'hit_rate@{k}'] for k in k_values]
        ndcg_values = [self.evaluation_results[f'ndcg@{k}'] for k in k_values]
        
        axes[0, 1].plot(k_values, hit_rate_values, 'mo-', label='Hit Rate@K', linewidth=2)
        axes[0, 1].plot(k_values, ndcg_values, 'co-', label='NDCG@K', linewidth=2)
        axes[0, 1].set_xlabel('K')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('🎯 Hit Rate และ NDCG @K', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Beyond Accuracy Metrics
        beyond_metrics = ['MRR', 'Diversity', 'Novelty', 'Coverage']
        beyond_values = [
            self.evaluation_results['mrr'],
            self.evaluation_results['diversity'],
            self.evaluation_results['novelty'],
            self.evaluation_results['catalog_coverage']
        ]
        
        bars = axes[1, 0].bar(beyond_metrics, beyond_values, 
                             color=['skyblue', 'lightgreen', 'orange', 'pink'])
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('🌈 Beyond Accuracy Metrics', fontweight='bold')
        axes[1, 0].set_ylim(0, 1)
        
        # เพิ่มค่าบนแท่งกราฟ
        for bar, value in zip(bars, beyond_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Metrics Comparison
        all_metrics = ['Precision@10', 'Recall@10', 'F1@10', 'Hit Rate@10', 'NDCG@10', 'MRR']
        all_values = [
            self.evaluation_results['precision@10'],
            self.evaluation_results['recall@10'],
            self.evaluation_results['f1@10'],
            self.evaluation_results['hit_rate@10'],
            self.evaluation_results['ndcg@10'],
            self.evaluation_results['mrr']
        ]
        
        bars = axes[1, 1].bar(all_metrics, all_values, color='steelblue')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('📊 Key Metrics Comparison', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # เพิ่มค่าบนแท่งกราฟ
        for bar, value in zip(bars, all_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, models_dict, user_item_matrix, df_menu, df_orders):
        """
        เปรียบเทียบโมเดลหลายตัว
        
        Parameters:
        models_dict: dictionary ของโมเดลต่างๆ {'model_name': model_instance}
        """
        comparison_results = {}
        
        print("🔍 เปรียบเทียบโมเดลต่างๆ...")
        
        for model_name, model in models_dict.items():
            print(f"\n📊 ประเมิน {model_name}...")
            evaluator = AdvancedRecommendationEvaluator(model)
            results = evaluator.evaluate_comprehensive(user_item_matrix, df_menu, df_orders)
            comparison_results[model_name] = results
        
        # สร้างตาราง comparison
        comparison_df = pd.DataFrame(comparison_results).T
        
        print("\n" + "="*80)
        print("📊 ตารางเปรียบเทียบโมเดล")
        print("="*80)
        print(comparison_df.round(4))
        
        # สร้างกราฟเปรียบเทียบ
        self._plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def _plot_model_comparison(self, comparison_df):
        """สร้างกราฟเปรียบเทียบโมเดล"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Precision@K comparison
        precision_cols = [col for col in comparison_df.columns if 'precision@' in col]
        comparison_df[precision_cols].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('📈 Precision@K Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. NDCG@K comparison
        ndcg_cols = [col for col in comparison_df.columns if 'ndcg@' in col]
        comparison_df[ndcg_cols].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('🎯 NDCG@K Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('NDCG')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Beyond Accuracy comparison
        beyond_cols = ['mrr', 'diversity', 'novelty', 'catalog_coverage']
        comparison_df[beyond_cols].plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('🌈 Beyond Accuracy Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Overall performance radar chart
        # สำหรับกราฟนี้ใช้ค่าเฉลี่ยของ metrics หลักๆ
        key_metrics = ['precision@10', 'recall@10', 'ndcg@10', 'mrr', 'diversity']
        if len(key_metrics) <= len(comparison_df.columns):
            avg_scores = comparison_df[key_metrics].mean(axis=1)
            avg_scores.plot(kind='bar', ax=axes[1, 1], color='steelblue')
            axes[1, 1].set_title('🏆 Overall Performance', fontweight='bold')
            axes[1, 1].set_ylabel('Average Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


class SimpleEvaluator:
    """คลาสสำหรับการประเมินแบบง่าย - backward compatibility"""
    
    def __init__(self, model):
        self.model = model
        self.advanced_evaluator = AdvancedRecommendationEvaluator(model)
    
    def evaluate_model(self, user_item_matrix, df_menu, df_orders):
        """ประเมินโมเดลแบบง่าย"""
        return self.advanced_evaluator.evaluate_comprehensive(
            user_item_matrix, df_menu, df_orders
        )
    
    def print_results(self):
        """แสดงผลการประเมิน"""
        self.advanced_evaluator.print_evaluation_results()
    
    def plot_results(self):
        """สร้างกราฟแสดงผล"""
        self.advanced_evaluator.plot_evaluation_metrics()
