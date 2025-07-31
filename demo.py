# 🎯 Demo Script - ทดสอบระบบแนะนำเมนู
"""
สคริปต์สำหรับทดสอบระบบแนะนำเมนูแบบง่ายๆ
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """โหลดข้อมูลที่สร้างไว้"""
    print("📂 กำลังโหลดข้อมูล...")
    
    customers = pd.read_csv('data/customers.csv')
    menu = pd.read_csv('data/menu.csv')
    orders = pd.read_csv('data/orders.csv')
    
    print(f"✅ โหลดข้อมูลเสร็จ: {len(customers)} ลูกค้า, {len(menu)} เมนู, {len(orders)} ออเดอร์")
    return customers, menu, orders

def create_simple_recommendation_system(customers, menu, orders):
    """สร้างระบบแนะนำแบบง่าย"""
    print("🧠 กำลังสร้างระบบแนะนำ...")
    
    # สร้าง User-Item Matrix
    interaction_matrix = orders.groupby(['customer_id', 'menu_id']).size().reset_index(name='interactions')
    user_item_matrix = interaction_matrix.pivot(
        index='customer_id', 
        columns='menu_id', 
        values='interactions'
    ).fillna(0)
    
    # ใช้ SVD
    svd = TruncatedSVD(n_components=10, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T
    predicted_ratings = np.dot(user_factors, item_factors.T)
    
    print("✅ สร้างระบบแนะนำเสร็จ")
    return user_item_matrix, predicted_ratings, menu

def recommend_for_customer(customer_id, user_item_matrix, predicted_ratings, menu, top_k=5):
    """แนะนำเมนูสำหรับลูกค้า"""
    try:
        user_idx = list(user_item_matrix.index).index(customer_id)
        
        # ดึงคะแนนทำนาย
        user_predictions = predicted_ratings[user_idx]
        
        # หาเมนูที่ยังไม่เคยสั่ง
        already_ordered = user_item_matrix.iloc[user_idx] > 0
        user_predictions[already_ordered] = -np.inf
        
        # หา top-k
        top_indices = np.argsort(user_predictions)[::-1][:top_k]
        menu_columns = user_item_matrix.columns
        recommended_menu_ids = [menu_columns[i] for i in top_indices]
        
        # ดึงข้อมูลเมนู
        recommendations = []
        for menu_id in recommended_menu_ids:
            menu_info = menu[menu['menu_id'] == menu_id].iloc[0]
            recommendations.append({
                'menu_name': menu_info['menu_name'],
                'category': menu_info['category'],
                'price': menu_info['price'],
                'popularity': menu_info['popularity']
            })
        
        return recommendations
        
    except ValueError:
        # ลูกค้าใหม่ - แนะนำเมนูยอดนิยม
        popular_menus = menu.nlargest(top_k, 'popularity')
        recommendations = []
        for _, menu_info in popular_menus.iterrows():
            recommendations.append({
                'menu_name': menu_info['menu_name'],
                'category': menu_info['category'],
                'price': menu_info['price'],
                'popularity': menu_info['popularity']
            })
        return recommendations

def demo_recommendations():
    """Demo การแนะนำเมนู"""
    print("🍽️ AI Menu Recommendation System - Demo")
    print("=" * 60)
    
    # โหลดข้อมูล
    customers, menu, orders = load_data()
    
    # สร้างระบบแนะนำ
    user_item_matrix, predicted_ratings, menu = create_simple_recommendation_system(customers, menu, orders)
    
    # ทดสอบกับลูกค้า 3 คนแรก
    test_customers = customers['customer_id'].head(3).tolist()
    
    for customer_id in test_customers:
        print(f"\n👤 ลูกค้า: {customer_id}")
        
        # ดูข้อมูลลูกค้า
        customer_info = customers[customers['customer_id'] == customer_id].iloc[0]
        print(f"   🎂 อายุ: {customer_info['age']} ปี")
        print(f"   ⚧ เพศ: {'ชาย' if customer_info['gender'] == 'M' else 'หญิง'}")
        print(f"   💰 งบประมาณ: {customer_info['avg_budget']:.2f} บาท")
        print(f"   ⏰ เวลาที่ชอบ: {customer_info['preferred_time']}")
        
        # ดูประวัติการสั่งอาหาร
        customer_orders = orders[orders['customer_id'] == customer_id]
        if len(customer_orders) > 0:
            favorite_menus = customer_orders['menu_name'].value_counts().head(3)
            print(f"   🍽️ เมนูที่ชอบ: {', '.join(favorite_menus.index.tolist())}")
        
        # ได้คำแนะนำ
        recommendations = recommend_for_customer(customer_id, user_item_matrix, predicted_ratings, menu)
        
        print(f"   🎯 เมนูที่แนะนำ:")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. {rec['menu_name']} ({rec['category']}) - {rec['price']:.2f} บาท ⭐{rec['popularity']}")
    
    print(f"\n🎉 Demo เสร็จสิ้น!")
    print(f"💡 เปิด Streamlit Dashboard: streamlit run dashboard/app.py")
    print(f"📊 เปิด Jupyter Notebook: jupyter notebook")

if __name__ == "__main__":
    demo_recommendations()
