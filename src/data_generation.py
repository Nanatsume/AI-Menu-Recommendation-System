"""
Data Generation Module for AI Menu Recommendation System
สร้างข้อมูลจำลองสำหรับระบบแนะนำเมนูอาหาร
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# ตั้งค่า random seed เพื่อผลลัพธ์ที่สม่ำเสมอ
np.random.seed(42)
random.seed(42)

class DataGenerator:
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # ข้อมูลพื้นฐานสำหรับการจำลอง
        self.thai_foods = [
            "ผัดไทย", "ข้าวผัด", "ต้มยำกุ้ง", "แกงเขียวหวานไก่", "ส้มตำ", "ลาบหมู", 
            "ผัดกะเพรา", "แกงส้มปลา", "ข้าวซอย", "หมี่กรอบ", "ปลากะพงนึ่งมะนาว",
            "แกงมัสมั่นเนื้อ", "ยำวุ้นเส้น", "ก๋วยเตี๋ยวต้มยำ", "ข้าวคลุกกะปิ",
            "ไก่ทอดหาดใหญ่", "หอยทอด", "ผัดซีอิ๊ว", "แกงเปอะ", "ข้าวกะเพรา"
        ]
        
        self.beverages = [
            "ชาเย็น", "กาแฟเย็น", "น้ำส้ม", "น้ำมะนาว", "โค้ก", "สไปรท์",
            "น้ำเปล่า", "ชาร้อน", "กาแฟร้อน", "น้ำผลไม้รวม", "น้ำแข็งใส",
            "น้ำอัดลม", "เบียร์", "น้ำแดง", "นมเย็น"
        ]
        
        self.desserts = [
            "ขนมครก", "ทับทิมกรอบ", "ข้าวเหนียวมะม่วง", "ลูกชุบ", "ฟักทอง",
            "บัวลอย", "ไอติม", "เค้ก", "คุกกี้", "บราวนี่", "ผลไม้"
        ]
        
        self.food_categories = {
            "อาหารหลัก": self.thai_foods,
            "เครื่องดื่ม": self.beverages, 
            "ของหวาน": self.desserts
        }
        
    def generate_customers(self, n_customers=500):
        """สร้างข้อมูลลูกค้า"""
        print(f"🧑‍🤝‍🧑 กำลังสร้างข้อมูลลูกค้า {n_customers} คน...")
        
        customers = []
        for i in range(n_customers):
            age = np.random.randint(18, 65)
            gender = random.choice(['M', 'F'])
            
            # กำหนดงบประมาณตามช่วงอายุ
            if age < 25:
                budget = np.random.normal(150, 50)
            elif age < 40:
                budget = np.random.normal(300, 100)
            else:
                budget = np.random.normal(250, 80)
            
            budget = max(50, budget)  # งบประมาณขั้นต่ำ 50 บาท
            
            # เวลาที่มักมาร้าน
            preferred_times = random.choices(
                ['morning', 'lunch', 'afternoon', 'dinner', 'late_night'],
                weights=[0.1, 0.3, 0.15, 0.35, 0.1]
            )[0]
            
            customers.append({
                'customer_id': f'C{i+1:04d}',
                'age': age,
                'gender': gender,
                'avg_budget': round(budget, 2),
                'preferred_time': preferred_times
            })
            
        df_customers = pd.DataFrame(customers)
        df_customers.to_csv(f'{self.data_dir}/customers.csv', index=False, encoding='utf-8-sig')
        print(f"✅ สร้างข้อมูลลูกค้าเสร็จสิ้น: {len(df_customers)} คน")
        return df_customers
    
    def generate_menu(self):
        """สร้างข้อมูลเมนูอาหาร"""
        print("🍽️ กำลังสร้างข้อมูลเมนู...")
        
        menu_items = []
        item_id = 1
        
        for category, foods in self.food_categories.items():
            for food in foods:
                # กำหนดราคาตามประเภทอาหาร
                if category == "อาหารหลัก":
                    price = np.random.normal(80, 30)
                elif category == "เครื่องดื่ม":
                    price = np.random.normal(25, 10)
                else:  # ของหวาน
                    price = np.random.normal(40, 15)
                
                price = max(15, round(price, 2))  # ราคาขั้นต่ำ 15 บาท
                
                # คะแนนความนิยม (1-5)
                popularity = round(np.random.uniform(2.5, 5.0), 1)
                
                menu_items.append({
                    'menu_id': f'M{item_id:03d}',
                    'menu_name': food,
                    'category': category,
                    'price': price,
                    'popularity': popularity
                })
                item_id += 1
        
        df_menu = pd.DataFrame(menu_items)
        df_menu.to_csv(f'{self.data_dir}/menu.csv', index=False, encoding='utf-8-sig')
        print(f"✅ สร้างข้อมูลเมนูเสร็จสิ้น: {len(df_menu)} เมนู")
        return df_menu
    
    def generate_orders(self, df_customers, df_menu, n_orders=10000):
        """สร้างประวัติการสั่งอาหาร"""
        print(f"📝 กำลังสร้างประวัติการสั่งอาหาร {n_orders} รายการ...")
        
        orders = []
        
        # สร้างข้อมูลการสั่งอาหารในช่วง 6 เดือนที่ผ่านมา
        start_date = datetime.now() - timedelta(days=180)
        
        for i in range(n_orders):
            # เลือกลูกค้าแบบสุ่ม
            customer = df_customers.sample(1).iloc[0]
            
            # สร้างวันที่สั่งอาหาร
            order_date = start_date + timedelta(days=random.randint(0, 180))
            
            # เลือกจำนวนเมนูที่สั่ง (1-4 เมนู)
            n_items = random.choices([1, 2, 3, 4], weights=[0.4, 0.35, 0.2, 0.05])[0]
            
            # เลือกเมนูตามความชอบ
            selected_menus = self._select_menus_for_customer(
                customer, df_menu, n_items
            )
            
            total_amount = 0
            for menu in selected_menus:
                quantity = random.choice([1, 1, 1, 2])  # ส่วนใหญ่สั่ง 1 จาน
                amount = menu['price'] * quantity
                total_amount += amount
                
                orders.append({
                    'order_id': f'O{i+1:06d}',
                    'customer_id': customer['customer_id'],
                    'menu_id': menu['menu_id'],
                    'menu_name': menu['menu_name'],
                    'category': menu['category'],
                    'quantity': quantity,
                    'unit_price': menu['price'],
                    'amount': amount,
                    'order_date': order_date.strftime('%Y-%m-%d'),
                    'order_time': self._get_order_time(customer['preferred_time']),
                    'day_of_week': order_date.strftime('%A')
                })
        
        df_orders = pd.DataFrame(orders)
        df_orders.to_csv(f'{self.data_dir}/orders.csv', index=False, encoding='utf-8-sig')
        print(f"✅ สร้างประวัติการสั่งอาหารเสร็จสิ้น: {len(df_orders)} รายการ")
        return df_orders
    
    def _select_menus_for_customer(self, customer, df_menu, n_items):
        """เลือกเมนูตามลักษณะของลูกค้า"""
        selected = []
        
        # กรองเมนูตามงบประมาณ
        affordable_menu = df_menu[df_menu['price'] <= customer['avg_budget'] * 0.8]
        
        if len(affordable_menu) == 0:
            affordable_menu = df_menu
        
        # เลือกเมนูตามความนิยมและความสุ่ม
        weights = affordable_menu['popularity'].values
        weights = weights / weights.sum()
        
        # เลือกเมนูไม่ซ้ำกัน
        selected_indices = np.random.choice(
            len(affordable_menu), 
            size=min(n_items, len(affordable_menu)), 
            replace=False, 
            p=weights
        )
        
        for idx in selected_indices:
            selected.append(affordable_menu.iloc[idx].to_dict())
        
        return selected
    
    def _get_order_time(self, preferred_time):
        """สร้างเวลาการสั่งอาหารตามช่วงเวลาที่ชอบ"""
        time_ranges = {
            'morning': (7, 10),
            'lunch': (11, 14), 
            'afternoon': (14, 17),
            'dinner': (17, 21),
            'late_night': (21, 23)
        }
        
        start_hour, end_hour = time_ranges[preferred_time]
        hour = random.randint(start_hour, end_hour)
        minute = random.randint(0, 59)
        
        return f"{hour:02d}:{minute:02d}"
    
    def generate_all_data(self, n_customers=500, n_orders=10000):
        """สร้างข้อมูลทั้งหมด"""
        print("🚀 เริ่มสร้างข้อมูลจำลองทั้งหมด...")
        
        # สร้างข้อมูลลูกค้า
        df_customers = self.generate_customers(n_customers)
        
        # สร้างข้อมูลเมนู
        df_menu = self.generate_menu()
        
        # สร้างประวัติการสั่งอาหาร
        df_orders = self.generate_orders(df_customers, df_menu, n_orders)
        
        print("\n📊 สรุปข้อมูลที่สร้าง:")
        print(f"   👥 ลูกค้า: {len(df_customers)} คน")
        print(f"   🍽️ เมนู: {len(df_menu)} เมนู")
        print(f"   📝 ออเดอร์: {len(df_orders)} รายการ")
        print(f"   💾 ไฟล์ถูกบันทึกใน folder: {self.data_dir}/")
        
        return df_customers, df_menu, df_orders

if __name__ == "__main__":
    # สร้างข้อมูลจำลอง
    generator = DataGenerator()
    df_customers, df_menu, df_orders = generator.generate_all_data()
    
    print("\n🎉 สร้างข้อมูลจำลองเสร็จสิ้นแล้ว!")
