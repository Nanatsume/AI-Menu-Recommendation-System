"""
Evaluation Module for AI Menu Recommendation System
ประเมินผลการทำงานของระบบแนะนำเมนู
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os

plt.rcParams['font.family'] = ['DejaVu Sans', 'Tahoma', 'SimHei']  # รองรับ Unicode

class RecommendationEvaluator:
    """คลาสสำหรับประเมินผลระบบแนะนำ"""
    
    def __init__(self, model_system, test_data_path="data/processed/test_data.csv"):
        self.system = model_system
        self.test_data = pd.read_csv(test_data_path)
        self.evaluation_results = {}
        
    def precision_at_k(self, recommended_items, relevant_items, k):
        """คำนวณ Precision@K"""
        if k == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = set(recommended_k) & set(relevant_items)
        
        return len(relevant_recommended) / min(k, len(recommended_items))
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        """คำนวณ Recall@K"""
        if len(relevant_items) == 0:
            return 0.0
            
        recommended_k = recommended_items[:k]
        relevant_recommended = set(recommended_k) & set(relevant_items)
        
        return len(relevant_recommended) / len(relevant_items)
    
    def hit_rate_at_k(self, recommended_items, relevant_items, k):
        """คำนวณ Hit Rate@K"""
        recommended_k = recommended_items[:k]
        return 1.0 if len(set(recommended_k) & set(relevant_items)) > 0 else 0.0
    
    def mean_reciprocal_rank(self, recommended_items, relevant_items):
        """คำนวณ Mean Reciprocal Rank (MRR)"""
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_recommendations(self, k_values=[5, 10, 20]):
        """ประเมินผลการแนะนำสำหรับผู้ใช้ทั้งหมด"""
        print("🧪 กำลังประเมินผลระบบแนะนำ...")
        
        # เตรียมข้อมูลการทดสอบ
        test_users = self.test_data[self.test_data['rating'] == 1]['customer_id'].unique()
        
        results = {k: {'precision': [], 'recall': [], 'hit_rate': [], 'mrr': []} for k in k_values}
        
        evaluated_users = 0
        for customer_id in test_users[:50]:  # ทดสอบ 50 คนแรก
            try:
                # หาเมนูที่ลูกค้าชอบจริงๆ ในชุดทดสอบ
                relevant_items = self.test_data[
                    (self.test_data['customer_id'] == customer_id) & 
                    (self.test_data['rating'] == 1)
                ]['menu_id'].values
                
                if len(relevant_items) == 0:
                    continue
                
                # ได้คำแนะนำจากระบบ
                recommendations = self.system.recommend_for_user(customer_id, top_k=max(k_values))
                recommended_items = [rec['menu_id'] for rec in recommendations]
                
                # คำนวณ metrics สำหรับแต่ละ k
                for k in k_values:
                    precision = self.precision_at_k(recommended_items, relevant_items, k)
                    recall = self.recall_at_k(recommended_items, relevant_items, k)
                    hit_rate = self.hit_rate_at_k(recommended_items, relevant_items, k)
                    
                    results[k]['precision'].append(precision)
                    results[k]['recall'].append(recall)
                    results[k]['hit_rate'].append(hit_rate)
                
                # MRR
                mrr = self.mean_reciprocal_rank(recommended_items, relevant_items)
                for k in k_values:
                    results[k]['mrr'].append(mrr)
                
                evaluated_users += 1
                
            except Exception as e:
                print(f"⚠️ ข้อผิดพลาดกับลูกค้า {customer_id}: {e}")
                continue
        
        # คำนวณค่าเฉลี่ย
        self.evaluation_results = {}
        for k in k_values:
            self.evaluation_results[k] = {
                'precision': np.mean(results[k]['precision']),
                'recall': np.mean(results[k]['recall']),
                'hit_rate': np.mean(results[k]['hit_rate']),
                'mrr': np.mean(results[k]['mrr']),
                'f1_score': 2 * np.mean(results[k]['precision']) * np.mean(results[k]['recall']) / 
                           (np.mean(results[k]['precision']) + np.mean(results[k]['recall'])) 
                           if (np.mean(results[k]['precision']) + np.mean(results[k]['recall'])) > 0 else 0
            }
        
        print(f"✅ ประเมินผลเสร็จสิ้น ({evaluated_users} ลูกค้า)")
        return self.evaluation_results
    
    def print_evaluation_results(self):
        """แสดงผลการประเมิน"""
        print("\n📊 ผลการประเมินระบบแนะนำ:")
        print("=" * 60)
        
        for k, metrics in self.evaluation_results.items():
            print(f"\n📈 Top-{k} Recommendations:")
            print(f"   🎯 Precision@{k}: {metrics['precision']:.4f}")
            print(f"   🔄 Recall@{k}: {metrics['recall']:.4f}")
            print(f"   ⚡ F1-Score@{k}: {metrics['f1_score']:.4f}")
            print(f"   🎪 Hit Rate@{k}: {metrics['hit_rate']:.4f}")
            print(f"   🏆 MRR@{k}: {metrics['mrr']:.4f}")
    
    def plot_evaluation_results(self, save_path="evaluation_results.png"):
        """สร้างกราฟแสดงผลการประเมิน"""
        print("📊 กำลังสร้างกราฟผลการประเมิน...")
        
        k_values = list(self.evaluation_results.keys())
        metrics = ['precision', 'recall', 'f1_score', 'hit_rate', 'mrr']
        metric_names = ['Precision', 'Recall', 'F1-Score', 'Hit Rate', 'MRR']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [self.evaluation_results[k][metric] for k in k_values]
            
            axes[i].plot(k_values, values, marker='o', linewidth=2, markersize=8)
            axes[i].set_title(f'{name}@K', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('K (Top-K)')
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, max(values) * 1.1 if max(values) > 0 else 1)
            
            # เพิ่มค่าบนจุด
            for j, v in enumerate(values):
                axes[i].annotate(f'{v:.3f}', (k_values[j], v), 
                               textcoords="offset points", xytext=(0,10), ha='center')
        
        # ลบ subplot ที่เหลือ
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ บันทึกกราฟแล้วที่: {save_path}")
    
    def analyze_category_performance(self):
        """วิเคราะห์ประสิทธิภาพตามหมวดหมู่เมนู"""
        print("📋 กำลังวิเคราะห์ประสิทธิภาพตามหมวดหมู่...")
        
        # โหลดข้อมูลเมนู
        menu_data = pd.read_csv("data/menu.csv")
        
        category_performance = {}
        test_users = self.test_data[self.test_data['rating'] == 1]['customer_id'].unique()[:20]
        
        for customer_id in test_users:
            try:
                recommendations = self.system.recommend_for_user(customer_id, top_k=10)
                
                for rec in recommendations:
                    menu_info = menu_data[menu_data['menu_id'] == rec['menu_id']]
                    if len(menu_info) > 0:
                        category = menu_info.iloc[0]['category']
                        
                        if category not in category_performance:
                            category_performance[category] = {'count': 0, 'total_score': 0}
                        
                        category_performance[category]['count'] += 1
                        category_performance[category]['total_score'] += rec['predicted_score']
            
            except Exception as e:
                continue
        
        # คำนวณค่าเฉลี่ย
        for category in category_performance:
            count = category_performance[category]['count']
            if count > 0:
                category_performance[category]['avg_score'] = (
                    category_performance[category]['total_score'] / count
                )
        
        print("\n📊 ประสิทธิภาพตามหมวดหมู่:")
        for category, perf in category_performance.items():
            if perf['count'] > 0:
                print(f"   {category}: {perf['avg_score']:.4f} (แนะนำ {perf['count']} ครั้ง)")
        
        return category_performance
    
    def diversity_analysis(self):
        """วิเคราะห์ความหลากหลายของการแนะนำ"""
        print("🌈 กำลังวิเคราะห์ความหลากหลาย...")
        
        test_users = self.test_data[self.test_data['rating'] == 1]['customer_id'].unique()[:20]
        all_recommendations = []
        
        for customer_id in test_users:
            try:
                recommendations = self.system.recommend_for_user(customer_id, top_k=10)
                recommended_items = [rec['menu_id'] for rec in recommendations]
                all_recommendations.extend(recommended_items)
            except:
                continue
        
        # คำนวณความหลากหलาย
        unique_items = len(set(all_recommendations))
        total_recommendations = len(all_recommendations)
        
        diversity_score = unique_items / total_recommendations if total_recommendations > 0 else 0
        
        print(f"🌟 Diversity Score: {diversity_score:.4f}")
        print(f"   📝 รายการที่แนะนำทั้งหมด: {total_recommendations}")
        print(f"   🎯 รายการที่ไม่ซ้ำ: {unique_items}")
        
        return diversity_score
    
    def create_recommendation_report(self, customer_id, save_path=None):
        """สร้างรายงานการแนะนำสำหรับลูกค้าคนหนึ่ง"""
        print(f"📄 กำลังสร้างรายงานสำหรับลูกค้า {customer_id}...")
        
        # ดึงข้อมูลลูกค้า
        profile = self.system.get_user_profile(customer_id)
        if profile is None:
            print("❌ ไม่พบข้อมูลลูกค้า")
            return
        
        # ได้คำแนะนำ
        recommendations = self.system.recommend_for_user(customer_id, top_k=10)
        
        # สร้างรายงาน
        report = f"""
🍽️ AI Menu Recommendation Report
=======================================

👤 ข้อมูลลูกค้า:
   🆔 รหัสลูกค้า: {profile['customer_id']}
   🎂 อายุ: {profile['age']} ปี
   💰 งบประมาณเฉลี่ย: {profile['avg_budget']:.2f} บาท
   📊 จำนวนออเดอร์: {profile['total_orders']} ครั้ง
   💵 ยอดเฉลี่ยต่อออเดอร์: {profile['avg_order_amount']:.2f} บาท

🏆 หมวดหมู่ที่ชื่นชอบ:
"""
        
        for category, count in profile['favorite_categories'].items():
            report += f"   {category}: {count} ครั้ง\n"
        
        report += "\n🍽️ เมนูที่แนะนำ (Top 10):\n"
        report += "=" * 50 + "\n"
        
        for i, rec in enumerate(recommendations, 1):
            report += f"{i:2d}. {rec['menu_name']} ({rec['category']})\n"
            report += f"     💰 ราคา: {rec['price']:.2f} บาท\n"
            report += f"     ⭐ คะแนนทำนาย: {rec['predicted_score']:.3f}\n"
            report += f"     📊 ความนิยม: {rec['popularity']:.1f}/5.0\n\n"
        
        # บันทึกรายงาน
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ บันทึกรายงานที่: {save_path}")
        
        print(report)
        return report
    
    def run_full_evaluation(self):
        """รันการประเมินผลแบบครบถ้วน"""
        print("🚀 เริ่มการประเมินผลแบบครบถ้วน...")
        
        # 1. ประเมินผลการแนะนำ
        self.evaluate_recommendations()
        self.print_evaluation_results()
        
        # 2. สร้างกราฟ
        self.plot_evaluation_results()
        
        # 3. วิเคราะห์ตามหมวดหมู่
        self.analyze_category_performance()
        
        # 4. วิเคราะห์ความหลากหลาย
        self.diversity_analysis()
        
        print("\n🎉 การประเมินผลเสร็จสิ้นแล้ว!")

if __name__ == "__main__":
    # โหลดโมเดลและรันการประเมินผล
    from models import HybridRecommendationSystem
    
    # สร้างระบบแนะนำ
    system = HybridRecommendationSystem()
    system.train_models()
    
    # ประเมินผล
    evaluator = RecommendationEvaluator(system)
    evaluator.run_full_evaluation()
    
    # สร้างรายงานตัวอย่าง
    evaluator.create_recommendation_report('C0001', 'sample_recommendation_report.txt')
