# 🍽️ AI Menu Recommendation System
# Quick Start Guide - การเริ่มต้นใช้งานด่วน

## ✅ ขั้นตอนการใช้งาน

### 1. รันครั้งแรก (สร้างข้อมูลและฝึกโมเดล)
```bash
python main.py --action all
```

### 2. รันเฉพาะส่วนที่ต้องการ
```bash
# สร้างข้อมูลเท่านั้น
python main.py --action generate

# เตรียมข้อมูลเท่านั้น  
python main.py --action preprocess

# ฝึกโมเดลเท่านั้น
python main.py --action train

# ประเมินผลเท่านั้น
python main.py --action evaluate

# ทดสอบการแนะนำ
python main.py --action demo

# เปิด Dashboard
python main.py --action dashboard
```

### 3. เปิด Jupyter Notebook
```bash
jupyter notebook
# หรือ
jupyter lab
```

### 4. เปิด Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

## 🚀 ทดสอบระบบ
ลองรันคำสั่งง่ายๆ เพื่อทดสอบว่าระบบทำงานได้:

```bash
python -c "print('🎉 Python ทำงานได้!')"
```

## 📝 หมายเหตุ
- ถ้าขาด packages ให้รัน: `pip install -r requirements.txt`
- ถ้ามีปัญหา import ให้ตรวจสอบ path และ environment
- Dashboard จะเปิดที่ http://localhost:8501
