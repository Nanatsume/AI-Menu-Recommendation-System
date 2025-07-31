# 🍽️ AI Menu Recommendation System

ระบบแนะนำเมนูอาหารอัจฉริยะสำหรับร้านอาหาร โดยใช้เทคนิค Machine Learning 

## 🎯 วัตถุประสงค์
- เพิ่มยอดขาย
- ลดเวลาในการตัดสินใจของลูกค้า  
- แนะนำเมนูที่เหมาะสมกับพฤติกรรมลูกค้าแบบเฉพาะบุคคล (Personalized)

## 📊 โครงสร้างข้อมูล
- **ลูกค้า**: เพศ, อายุ, เวลาที่มาร้าน, งบประมาณเฉลี่ย, ประวัติการสั่งอาหาร
- **เมนูอาหาร**: เมนู, หมวดหมู่, ราคา, ความนิยม
- **ข้อมูลเสริม**: วันในสัปดาห์, เทศกาล, ฤดูกาล

## 🧠 เทคนิค AI ที่ใช้
- **Preprocessing**: Pandas, Scikit-learn
- **Model**: Collaborative Filtering, Neural Collaborative Filtering
- **Evaluation**: Precision@K, Recall@K, Hit Rate
- **Visualization**: Matplotlib, Seaborn
- **Dashboard**: Streamlit

## 📁 โครงสร้างโปรเจค
```
├── data/                   # ข้อมูลจำลอง
├── src/                    # โค้ดหลัก
│   ├── data_generation.py  # สร้างข้อมูลจำลอง
│   ├── preprocessing.py    # เตรียมข้อมูล
│   ├── models.py          # โมเดล ML
│   └── evaluation.py      # ประเมินผล
├── notebooks/             # Jupyter notebooks
├── dashboard/             # Streamlit dashboard
├── requirements.txt       # dependencies
└── main.py               # รันระบบหลัก
```

## 🚀 การใช้งาน
1. สร้างข้อมูลจำลอง: `python src/data_generation.py`
2. ฝึกโมเดล: `python main.py`
3. เปิด Dashboard: `streamlit run dashboard/app.py`

## 📈 ผลลัพธ์ที่คาดหวัง
- ระบบสามารถแนะนำเมนูที่เหมาะสมได้อย่างแม่นยำ
- เพิ่มโอกาสการขายเมนูใหม่ๆ
- ลูกค้าประหยัดเวลาในการเลือกเมนู
