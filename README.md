# AI Menu Recommendation System

ระบบแนะนำเมนูอาหารอัจฉริยะสำหรับร้านอาหาร โดยใช้เทคนิค Machine Learning และ Deep Learning

## วัตถุประสงค์
- **เพิ่มยอดขาย** ด้วยการแนะนำเมนูที่ตรงใจลูกค้า
- **ลดเวลาในการตัดสินใจ** ของลูกค้าในการเลือกเมนู
- **Personalized Recommendations** แนะนำเฉพาะบุคคลตามพฤติกรรม
- **Business Intelligence** วิเคราะห์ข้อมูลเพื่อการตัดสินใจทางธุรกิจ

## เทคนิค AI ที่ใช้
- **Matrix Factorization** (SVD) - Collaborative Filtering
- **Neural Collaborative Filtering** - Deep Learning
- **Hybrid Recommendation System** - รวมหลายเทคนิค
- **Content-Based Filtering** - ใช้ข้อมูลเมนู
- **Customer Segmentation** - K-means Clustering
- **Advanced Evaluation** - 15+ metrics

## โครงสร้างข้อมูล
- **ลูกค้า**: 500 คน (เพศ, อายุ, งบประมาณ, เวลาที่ชอบ)
- **เมนูอาหาร**: 46 เมนู (3 หมวดหมู่: อาหารหลัก, เครื่องดื่ม, ของหวาน)
- **ประวัติการสั่ง**: 18,000+ รายการ พร้อมคะแนนความพึงพอใจ

## โครงสร้างโปรเจค
```
├── data/                        # ข้อมูลหลัก
│   ├── customers.csv               # ข้อมูลลูกค้า
│   ├── menu.csv                    # ข้อมูลเมนู  
│   ├── orders.csv                  # ข้อมูลการสั่งอาหาร
│   └── processed/                  # ข้อมูลที่เตรียมแล้ว
├── src/                         # โค้ดหลัก AI/ML
│   ├── data_generation.py          # สร้างข้อมูลจำลอง
│   ├── preprocessing.py            # เตรียมข้อมูล
│   ├── models.py                   # โมเดล ML หลัก
│   ├── model_factory.py            # Factory Pattern
│   ├── evaluation_fixed.py         # ประเมินผล 15+ metrics
│   └── advanced_features.py        # ฟีเจอร์ขั้นสูง
├── notebooks/                   # การวิเคราะห์
│   └── AI-Recommend.ipynb          # EDA & Experiments
├── dashboard/                   # Streamlit Dashboard
│   └── app.py                      # แดชบอร์ดหลัก
├── app/                         # Web Application  
│   └── streamlit_app.py            # เว็บแอปใหม่
├── main.py                      # รันระบบหลัก
├── demo.py                      # ทดสอบการแนะนำ
└── requirements.txt             # dependencies
```

## การใช้งาน

### Quick Start
```bash
# รันครั้งแรก (ทำทุกอย่าง)
python main.py --action all

# เปิด Dashboard
streamlit run dashboard/app.py

# เปิด Web App (แนะนำ)
streamlit run app/streamlit_app.py
```

### รันแยกส่วน
```bash
# สร้างข้อมูล
python main.py --action generate

# เตรียมข้อมูล
python main.py --action preprocess  

# ฝึกโมเดล
python main.py --action train

# ประเมินผล
python main.py --action evaluate

# ทดสอบการแนะนำ
python demo.py
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/AI-Recommend.ipynb
```

## ฟีเจอร์หลัก

### ระบบแนะนำ
- **Matrix Factorization**: ลด sparsity ด้วย SVD
- **Neural CF**: Deep Learning สำหรับ pattern ซับซ้อน
- **Hybrid System**: รวมจุดแข็งหลายโมเดล
- **Content-Based**: ใช้ข้อมูลเมนู (ราคา, หมวดหมู่)

### การประเมินผล
- **Accuracy**: Precision@K, Recall@K, NDCG@K
- **Beyond Accuracy**: Diversity, Coverage, Novelty
- **Business Metrics**: Hit Rate, MRR
- **A/B Testing**: เปรียบเทียบโมเดล

### Customer Intelligence
- **Segmentation**: แบ่งกลุ่มลูกค้า 4 กลุ่ม (K-Means)
- **Trend Analysis**: วิเคราะห์เทรนด์ตามเวลา
- **Business Intelligence**: รายงานทางธุรกิจ

### User Interface
- **Streamlit Dashboard**: แดชบอร์ดสวยงาม
- **Web Application**: ทดสอบการแนะนำแบบ Interactive
- **Jupyter Notebook**: การวิเคราะห์เชิงลึก

## ผลลัพธ์ที่คาดหวัง
- **เพิ่มยอดขาย 15-25%** จากการแนะนำที่แม่นยำ
- **ลดเวลาตัดสินใจ 30-40%** ของลูกค้า
- **เพิ่มความพึงพอใจลูกค้า** จาก personalization
- **ปรับปรุงการจัดการสต็อก** จากการพยากรณ์ความต้องการ

## เทคโนโลยีที่ใช้
- **Python 3.9+** - ภาษาหลัก
- **scikit-learn** - Machine Learning
- **TensorFlow/Keras** - Deep Learning (Neural CF)
- **pandas, numpy** - การจัดการข้อมูล
- **Streamlit** - Web Dashboard
- **Plotly** - Data Visualization
- **Jupyter** - การวิเคราะห์

## Requirements
```bash
pip install -r requirements.txt
```

**หรือติดตั้งแบบ manual:**
```bash
pip install streamlit pandas numpy scikit-learn tensorflow plotly seaborn matplotlib
```

### สิ่งที่เสร็จแล้ว:
- Advanced AI Models (6 algorithms)
- Comprehensive Evaluation (15+ metrics)  
- Customer Segmentation & Analysis
- Interactive Web Applications (2 apps)
- Business Intelligence Dashboard
- A/B Testing Framework
- Complete Documentation

### การพัฒนาต่อ:
- Deep Learning ขั้นสูง (Graph Neural Networks)
- Real-time Recommendations API
- Multi-armed Bandit Optimization
- Cloud Deployment (Docker + Kubernetes)

## License
MIT License - ใช้งานได้อย่างอิสระ

## ติดต่อ

### **ผู้พัฒนา**
- **GitHub:** [@Nanatsume](https://github.com/Nanatsume)
- **Email:** ntphototh@gmail.com
- **LinkedIn:** [Nhatthapong](https://www.linkedin.com/in/nhatthapong-pukdeeboon-205203369/)

### **การสนับสนุน**
- **Bug Reports:** [Issues](https://github.com/Nanatsume/hr-optimization-system/issues)
- **Feature Requests:** [Discussions](https://github.com/Nanatsume/hr-optimization-system/discussions)
- **Documentation:** [Wiki](https://github.com/Nanatsume/hr-optimization-system/wiki)
---

*เวอร์ชัน: 2.0 | อัพเดตล่าสุด: August 2025*
