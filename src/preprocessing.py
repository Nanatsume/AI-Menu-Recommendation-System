"""
Data Preprocessing Module for AI Menu Recommendation System
เตรียมข้อมูลสำหรับการฝึกโมเดล
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class DataPreprocessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, df_customers=None, df_menu=None, df_orders=None):
        """โหลดข้อมูลจากไฟล์ CSV หรือ DataFrame"""
        print("📂 กำลังโหลดข้อมูล...")
        
        if df_customers is not None and df_menu is not None and df_orders is not None:
            # ใช้ DataFrame ที่ส่งมา
            self.df_customers = df_customers.copy()
            self.df_menu = df_menu.copy()
            self.df_orders = df_orders.copy()
            
            print(f"✅ โหลดข้อมูลจาก DataFrame เสร็จสิ้น:")
            print(f"   👥 ลูกค้า: {len(self.df_customers)} คน")
            print(f"   🍽️ เมนู: {len(self.df_menu)} เมนู")
            print(f"   📝 ออเดอร์: {len(self.df_orders)} รายการ")
            
        else:
            # โหลดจากไฟล์ CSV
            try:
                self.df_customers = pd.read_csv(f'{self.data_dir}/customers.csv')
                self.df_menu = pd.read_csv(f'{self.data_dir}/menu.csv') 
                self.df_orders = pd.read_csv(f'{self.data_dir}/orders.csv')
                
                print(f"✅ โหลดข้อมูลจากไฟล์เสร็จสิ้น:")
                print(f"   👥 ลูกค้า: {len(self.df_customers)} คน")
                print(f"   🍽️ เมนู: {len(self.df_menu)} เมนู")
                print(f"   📝 ออเดอร์: {len(self.df_orders)} รายการ")
                
            except FileNotFoundError as e:
                print(f"❌ ไม่พบไฟล์ข้อมูล: {e}")
                print("💡 กรุณารันไฟล์ data_generation.py ก่อน")
                return False
            
        return True
    
    def create_user_item_matrix(self):
        """สร้าง User-Item Matrix สำหรับ Collaborative Filtering"""
        print("🔢 กำลังสร้าง User-Item Matrix...")
        
        # นับจำนวนครั้งที่ลูกค้าแต่ละคนสั่งเมนูแต่ละอย่าง
        interaction_matrix = self.df_orders.groupby(['customer_id', 'menu_id']).size().reset_index(name='interactions')
        
        # สร้าง pivot table
        self.user_item_matrix = interaction_matrix.pivot(
            index='customer_id', 
            columns='menu_id', 
            values='interactions'
        ).fillna(0)
        
        print(f"✅ สร้าง User-Item Matrix เสร็จสิ้น: {self.user_item_matrix.shape}")
        
        # สร้าง binary matrix (1 = เคยสั่ง, 0 = ไม่เคยสั่ง)
        self.binary_matrix = (self.user_item_matrix > 0).astype(int)
        
        return self.user_item_matrix, self.binary_matrix
    
    def prepare_training_data(self):
        """เตรียมข้อมูลสำหรับการฝึกโมเดล (alias สำหรับ create_training_data)"""
        # เตรียม features ก่อน
        if not hasattr(self, 'user_features'):
            self.prepare_features()
        return self.create_training_data()
    
    def calculate_sparsity(self):
        """คำนวณค่า sparsity ของ User-Item Matrix"""
        if not hasattr(self, 'user_item_matrix'):
            print("⚠️ ยังไม่ได้สร้าง User-Item Matrix")
            return None
            
        total_possible_ratings = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        actual_ratings = np.count_nonzero(self.user_item_matrix.values)
        sparsity = (1 - (actual_ratings / total_possible_ratings)) * 100
        
        return sparsity
    
    def prepare_features(self):
        """เตรียม features สำหรับโมเดล"""
        print("🔧 กำลังเตรียม features...")
        
        # Encode categorical variables
        categorical_cols = ['gender', 'preferred_time']
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            self.df_customers[f'{col}_encoded'] = self.encoders[col].fit_transform(self.df_customers[col])
        
        # Menu features
        menu_categorical_cols = ['category']
        for col in menu_categorical_cols:
            self.encoders[f'menu_{col}'] = LabelEncoder()
            self.df_menu[f'{col}_encoded'] = self.encoders[f'menu_{col}'].fit_transform(self.df_menu[col])
        
        # สร้าง user features
        self.user_features = self.df_customers[['customer_id', 'age', 'avg_budget', 
                                               'gender_encoded', 'preferred_time_encoded']].copy()
        
        # สร้าง item features  
        self.item_features = self.df_menu[['menu_id', 'price', 'popularity', 
                                          'category_encoded']].copy()
        
        print("✅ เตรียม features เสร็จสิ้น")
        
        return self.user_features, self.item_features
    
    def create_training_data(self, test_size=0.2):
        """สร้างข้อมูลสำหรับการฝึกโมเดล"""
        print("📚 กำลังสร้างข้อมูลสำหรับฝึกโมเดล...")
        
        # สร้าง positive samples จากการสั่งอาหารจริง
        positive_samples = []
        for _, row in self.df_orders.iterrows():
            positive_samples.append({
                'customer_id': row['customer_id'],
                'menu_id': row['menu_id'],
                'rating': 1  # เคยสั่ง = 1
            })
        
        # สร้าง negative samples (เมนูที่ไม่เคยสั่ง)
        negative_samples = []
        all_customers = set(self.df_customers['customer_id'])
        all_menus = set(self.df_menu['menu_id'])
        
        # สำหรับแต่ละลูกค้า สร้าง negative samples
        for customer_id in all_customers:
            # เมนูที่เคยสั่ง
            ordered_menus = set(self.df_orders[self.df_orders['customer_id'] == customer_id]['menu_id'])
            # เมนูที่ไม่เคยสั่ง
            not_ordered = all_menus - ordered_menus
            
            # สุ่มเลือก negative samples (จำนวนเท่ากับ positive samples)
            n_negative = len(ordered_menus)
            if len(not_ordered) >= n_negative:
                sampled_negative = np.random.choice(list(not_ordered), n_negative, replace=False)
                for menu_id in sampled_negative:
                    negative_samples.append({
                        'customer_id': customer_id,
                        'menu_id': menu_id,
                        'rating': 0  # ไม่เคยสั่ง = 0
                    })
        
        # รวม positive และ negative samples
        all_samples = positive_samples + negative_samples
        self.training_data = pd.DataFrame(all_samples)
        
        # เพิ่ม features
        self.training_data = self.training_data.merge(
            self.user_features, on='customer_id', how='left'
        ).merge(
            self.item_features, on='menu_id', how='left'
        )
        
        # แบ่งข้อมูลเป็น train/test
        self.train_data, self.test_data = train_test_split(
            self.training_data, test_size=test_size, random_state=42, stratify=self.training_data['rating']
        )
        
        print(f"✅ สร้างข้อมูลฝึกโมเดลเสร็จสิ้น:")
        print(f"   📚 Train: {len(self.train_data)} samples")
        print(f"   🧪 Test: {len(self.test_data)} samples")
        print(f"   ➕ Positive: {sum(self.training_data['rating'])} samples")
        print(f"   ➖ Negative: {len(self.training_data) - sum(self.training_data['rating'])} samples")
        
        return self.train_data, self.test_data
    
    def create_user_item_mappings(self):
        """สร้าง mapping dictionaries สำหรับการแปลง ID"""
        print("🗺️ กำลังสร้าง ID mappings...")
        
        # User mappings
        unique_users = sorted(self.df_customers['customer_id'].unique())
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        
        # Item mappings  
        unique_items = sorted(self.df_menu['menu_id'].unique())
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        print(f"✅ สร้าง mappings เสร็จสิ้น: {len(unique_users)} users, {len(unique_items)} items")
        
        return self.user_to_idx, self.item_to_idx
    
    def save_preprocessed_data(self, output_dir="data/processed"):
        """บันทึกข้อมูลที่เตรียมแล้ว"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 กำลังบันทึกข้อมูลที่เตรียมแล้วใน {output_dir}/...")
        
        # บันทึก DataFrames
        self.train_data.to_csv(f'{output_dir}/train_data.csv', index=False)
        self.test_data.to_csv(f'{output_dir}/test_data.csv', index=False)
        self.user_features.to_csv(f'{output_dir}/user_features.csv', index=False)
        self.item_features.to_csv(f'{output_dir}/item_features.csv', index=False)
        
        # บันทึก matrices
        np.save(f'{output_dir}/user_item_matrix.npy', self.user_item_matrix.values)
        np.save(f'{output_dir}/binary_matrix.npy', self.binary_matrix.values)
        
        # บันทึก mappings และ encoders
        with open(f'{output_dir}/mappings.pkl', 'wb') as f:
            pickle.dump({
                'user_to_idx': self.user_to_idx,
                'item_to_idx': self.item_to_idx,
                'idx_to_user': self.idx_to_user,
                'idx_to_item': self.idx_to_item,
                'user_ids': list(self.user_item_matrix.index),
                'item_ids': list(self.user_item_matrix.columns)
            }, f)
        
        with open(f'{output_dir}/encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        print("✅ บันทึกข้อมูลเสร็จสิ้น")
    
    def get_data_summary(self):
        """แสดงสรุปข้อมูล"""
        print("\n📊 สรุปข้อมูลหลังการเตรียม:")
        print(f"   👥 จำนวนลูกค้า: {len(self.df_customers)}")
        print(f"   🍽️ จำนวนเมนู: {len(self.df_menu)}")
        print(f"   📝 จำนวนออเดอร์: {len(self.df_orders)}")
        print(f"   🔢 Matrix shape: {self.user_item_matrix.shape}")
        print(f"   📚 Training samples: {len(self.training_data)}")
        
        # แสดงการกระจายตัวของหมวดหมู่เมนู
        print(f"\n📋 การกระจายตัวของหมวดหมู่เมนู:")
        category_counts = self.df_menu['category'].value_counts()
        for category, count in category_counts.items():
            print(f"   {category}: {count} เมนู")
        
        # แสดงการกระจายตัวของช่วงอายุลูกค้า
        print(f"\n👥 การกระจายตัวของอายุลูกค้า:")
        age_bins = pd.cut(self.df_customers['age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        age_counts = age_bins.value_counts()
        for age_range, count in age_counts.items():
            print(f"   {age_range}: {count} คน")
    
    def run_preprocessing(self):
        """รันกระบวนการ preprocessing ทั้งหมด"""
        print("🚀 เริ่มกระบวนการเตรียมข้อมูล...")
        
        # โหลดข้อมูล
        if not self.load_data():
            return False
        
        # สร้าง User-Item Matrix
        self.create_user_item_matrix()
        
        # เตรียม features
        self.prepare_features()
        
        # สร้างข้อมูลฝึกโมเดล
        self.create_training_data()
        
        # สร้าง mappings
        self.create_user_item_mappings()
        
        # บันทึกข้อมูล
        self.save_preprocessed_data()
        
        # แสดงสรุป
        self.get_data_summary()
        
        print("\n🎉 เตรียมข้อมูลเสร็จสิ้น!")
        return True

if __name__ == "__main__":
    # รันการเตรียมข้อมูล
    preprocessor = DataPreprocessor()
    preprocessor.run_preprocessing()
