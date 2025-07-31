"""
Main Application for AI Menu Recommendation System
แอปพลิเคชันหลักสำหรับระบบแนะนำเมนูอาหาร
"""

import os
import sys
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import DataGenerator
from preprocessing import DataPreprocessor
from models import HybridRecommendationSystem
from evaluation import RecommendationEvaluator

def print_banner():
    """แสดง banner ของระบบ"""
    banner = """
    🍽️ =============================================== 🍽️
    
         AI Menu Recommendation System
         ระบบแนะนำเมนูอาหารอัจฉริยะ
    
    🍽️ =============================================== 🍽️
    """
    print(banner)

def generate_data():
    """สร้างข้อมูลจำลอง"""
    print("🚀 เริ่มสร้างข้อมูลจำลอง...")
    
    generator = DataGenerator()
    df_customers, df_menu, df_orders = generator.generate_all_data(
        n_customers=500, 
        n_orders=10000
    )
    
    print("✅ สร้างข้อมูลจำลองเสร็จสิ้น!")
    return True

def preprocess_data():
    """เตรียมข้อมูลสำหรับการฝึกโมเดล"""
    print("🔧 เริ่มกระบวนการเตรียมข้อมูล...")
    
    preprocessor = DataPreprocessor()
    success = preprocessor.run_preprocessing()
    
    if success:
        print("✅ เตรียมข้อมูลเสร็จสิ้น!")
    else:
        print("❌ เกิดข้อผิดพลาดในการเตรียมข้อมูล!")
    
    return success

def train_models():
    """ฝึกโมเดล AI"""
    print("🧠 เริ่มฝึกโมเดล AI...")
    
    try:
        system = HybridRecommendationSystem()
        system.train_models()
        system.save_models()
        
        print("✅ ฝึกโมเดลเสร็จสิ้น!")
        return system
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการฝึกโมเดล: {e}")
        return None

def evaluate_models(system):
    """ประเมินผลโมเดล"""
    print("📊 เริ่มประเมินผลโมเดล...")
    
    try:
        evaluator = RecommendationEvaluator(system)
        evaluator.run_full_evaluation()
        
        print("✅ ประเมินผลเสร็จสิ้น!")
        return True
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการประเมินผล: {e}")
        return False

def demo_recommendations(system):
    """แสดงตัวอย่างการแนะนำ"""
    print("🎯 ตัวอย่างการแนะนำเมนู:")
    print("=" * 50)
    
    # ทดสอบกับลูกค้า 3 คนแรก
    test_customers = ['C0001', 'C0002', 'C0003']
    
    for customer_id in test_customers:
        try:
            print(f"\n👤 ลูกค้า: {customer_id}")
            
            # ดึงโปรไฟล์ลูกค้า
            profile = system.get_user_profile(customer_id)
            if profile:
                print(f"   🎂 อายุ: {profile['age']} ปี")
                print(f"   💰 งบประมาณเฉลี่ย: {profile['avg_budget']:.2f} บาท")
                print(f"   📊 จำนวนออเดอร์: {profile['total_orders']} ครั้ง")
            
            # ได้คำแนะนำ
            recommendations = system.recommend_for_user(customer_id, top_k=5)
            
            print(f"   🍽️ เมนูที่แนะนำ:")
            for i, rec in enumerate(recommendations, 1):
                print(f"      {i}. {rec['menu_name']} ({rec['category']}) - {rec['price']:.2f} บาท")
                print(f"         คะแนน: {rec['predicted_score']:.3f}")
            
        except Exception as e:
            print(f"   ❌ ข้อผิดพลาด: {e}")
    
    print("\n" + "=" * 50)

def run_dashboard():
    """เปิด Streamlit Dashboard"""
    print("🌐 เปิด Dashboard...")
    print("📱 เปิดบราวเซอร์และไปที่: http://localhost:8501")
    print("⏹️ กด Ctrl+C เพื่อหยุดการทำงาน")
    
    os.system("streamlit run dashboard/app.py")

def install_dependencies():
    """ติดตั้ง dependencies"""
    print("📦 กำลังติดตั้ง dependencies...")
    
    os.system("pip install -r requirements.txt")
    
    print("✅ ติดตั้ง dependencies เสร็จสิ้น!")

def main():
    """ฟังก์ชันหลัก"""
    print_banner()
    
    parser = argparse.ArgumentParser(description='AI Menu Recommendation System')
    parser.add_argument('--action', choices=[
        'install', 'generate', 'preprocess', 'train', 'evaluate', 'demo', 'dashboard', 'all'
    ], default='all', help='เลือกการทำงาน')
    
    args = parser.parse_args()
    action = args.action
    
    print(f"🎯 การทำงานที่เลือก: {action}")
    print(f"⏰ เวลาเริ่มต้น: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        if action == 'install':
            install_dependencies()
            
        elif action == 'generate':
            generate_data()
            
        elif action == 'preprocess':
            if not os.path.exists('data/customers.csv'):
                print("⚠️ ไม่พบข้อมูล กำลังสร้างข้อมูลจำลองก่อน...")
                generate_data()
            preprocess_data()
            
        elif action == 'train':
            if not os.path.exists('data/processed/train_data.csv'):
                print("⚠️ ไม่พบข้อมูลที่เตรียมแล้ว กำลังเตรียมข้อมูลก่อน...")
                if not os.path.exists('data/customers.csv'):
                    generate_data()
                preprocess_data()
            train_models()
            
        elif action == 'evaluate':
            if not os.path.exists('models/matrix_model.pkl'):
                print("⚠️ ไม่พบโมเดลที่ฝึกแล้ว กำลังฝึกโมเดลก่อน...")
                if not os.path.exists('data/processed/train_data.csv'):
                    if not os.path.exists('data/customers.csv'):
                        generate_data()
                    preprocess_data()
                system = train_models()
            else:
                system = HybridRecommendationSystem()
            
            if system:
                evaluate_models(system)
            
        elif action == 'demo':
            if not os.path.exists('models/matrix_model.pkl'):
                print("⚠️ ไม่พบโมเดลที่ฝึกแล้ว กำลังฝึกโมเดลก่อน...")
                if not os.path.exists('data/processed/train_data.csv'):
                    if not os.path.exists('data/customers.csv'):
                        generate_data()
                    preprocess_data()
                system = train_models()
            else:
                system = HybridRecommendationSystem()
            
            if system:
                demo_recommendations(system)
                
        elif action == 'dashboard':
            run_dashboard()
            
        elif action == 'all':
            # รันทุกขั้นตอน
            print("🚀 รันระบบแบบสมบูรณ์...")
            
            # 1. สร้างข้อมูล (ถ้ายังไม่มี)
            if not os.path.exists('data/customers.csv'):
                generate_data()
            else:
                print("✅ พบข้อมูลที่มีอยู่แล้ว")
            
            # 2. เตรียมข้อมูล (ถ้ายังไม่มี)
            if not os.path.exists('data/processed/train_data.csv'):
                preprocess_data()
            else:
                print("✅ พบข้อมูลที่เตรียมแล้ว")
            
            # 3. ฝึกโมเดล (ถ้ายังไม่มี)
            if not os.path.exists('models/matrix_model.pkl'):
                system = train_models()
            else:
                print("✅ พบโมเดลที่ฝึกแล้ว")
                system = HybridRecommendationSystem()
            
            # 4. ประเมินผล
            if system:
                evaluate_models(system)
                demo_recommendations(system)
            
            # 5. เปิด dashboard
            print("\n🌐 พร้อมเปิด Dashboard!")
            print("💡 รันคำสั่ง: python main.py --action dashboard")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ หยุดการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n⏰ เวลาสิ้นสุด: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🙏 ขอบคุณที่ใช้ AI Menu Recommendation System!")

if __name__ == "__main__":
    main()
