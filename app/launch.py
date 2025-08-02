"""
🚀 Quick Launch Script for AI Menu Recommendation Web App
สคริปต์สำหรับเปิดเว็บแอปพลิเคชันอย่างรวดเร็ว
"""

import subprocess
import sys
import os
import time

def install_requirements():
    """ติดตั้ง dependencies ที่จำเป็น"""
    print("📦 กำลังติดตั้ง dependencies...")
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    for package in required_packages:
        try:
            print(f"   กำลังติดตั้ง {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            print(f"   ✅ {package} ติดตั้งสำเร็จ")
        except subprocess.CalledProcessError:
            print(f"   ⚠️ ไม่สามารถติดตั้ง {package} ได้")

def check_dependencies():
    """ตรวจสอบ dependencies"""
    print("\n🔍 ตรวจสอบ dependencies...")
    
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        import sklearn
        print("✅ Dependencies พร้อมใช้งาน!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def launch_app():
    """เปิดเว็บแอปพลิเคชัน"""
    print("\n🚀 กำลังเปิดเว็บแอปพลิเคชัน...")
    print("   📍 URL: http://localhost:8501")
    print("   🔄 กดปุ่ม Ctrl+C เพื่อหยุด")
    print("   🌐 เว็บเบราว์เซอร์จะเปิดอัตโนมัติ")
    
    try:
        # เปลี่ยน directory ไปที่ app folder
        app_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(app_dir)
        
        # รัน Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], 
                      check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 แอปพลิเคชันถูกหยุดโดยผู้ใช้")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching app: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def main():
    """ฟังก์ชันหลัก"""
    print("="*60)
    print("🍽️ AI Menu Recommendation System - Quick Launcher")
    print("="*60)
    
    # ตรวจสอบ dependencies
    if not check_dependencies():
        install_choice = input("\n❓ ต้องการติดตั้ง dependencies หรือไม่? (y/n): ")
        if install_choice.lower() in ['y', 'yes', 'ใช่']:
            install_requirements()
            if not check_dependencies():
                print("❌ ไม่สามารถติดตั้ง dependencies ได้ กรุณาติดตั้งด้วยตนเอง")
                return
        else:
            print("⚠️ ไม่สามารถเปิดแอปได้ เนื่องจากขาด dependencies")
            return
    
    # แสดงข้อมูลแอป
    print("\n📱 ฟีเจอร์ของเว็บแอปพลิเคชัน:")
    print("   🏠 หน้าหลัก - ภาพรวมระบบ")
    print("   🤖 ทดสอบระบบแนะนำ - AI Recommendations")
    print("   👥 วิเคราะห์ลูกค้า - Customer Segmentation")
    print("   📊 Business Intelligence - Dashboard")
    print("   📈 Model Performance - AI Metrics")
    
    # ถามผู้ใช้
    launch_choice = input("\n❓ ต้องการเปิดเว็บแอปพลิเคชันหรือไม่? (y/n): ")
    if launch_choice.lower() in ['y', 'yes', 'ใช่']:
        launch_app()
    else:
        print("👋 ขอบคุณที่ใช้งาน AI Menu Recommendation System!")

if __name__ == "__main__":
    main()
