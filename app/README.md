# 🍽️ AI Menu Recommendation System - Web Application

## วิธีการรันเว็บแอปพลิเคชัน

### 1. ติดตั้ง Dependencies
```bash
pip install streamlit plotly
```

### 2. รันแอปพลิเคชัน
```bash
# ใน terminal
cd "e:\Deep Learning\app"
streamlit run streamlit_app.py
```

### 3. เปิดเว็บเบราว์เซอร์
- แอปจะเปิดที่ `http://localhost:8501`
- หรือ Streamlit จะเปิดให้อัตโนมัติ

---

## ฟีเจอร์ของเว็บแอปพลิเคชัน

### 🏠 หน้าหลัก (Homepage)
- ภาพรวมระบบและสถิติทั่วไป
- กราฟแสดงยอดขายตามหมวดหมู่
- เมนูยอดนิยม Top 10
- กิจกรรมล่าสุด

### 🤖 ทดสอบระบบแนะนำ (Recommendation Testing)
- เลือกลูกค้าเพื่อทดสอบ
- แสดงข้อมูลลูกค้า (อายุ, เพศ, งบประมาณ)
- สร้างคำแนะนำเมนูด้วย AI
- แสดงประวัติการสั่งอาหาร
- ปรับจำนวนเมนูที่แนะนำได้

### 👥 วิเคราะห์ลูกค้า (Customer Analysis)
- Customer Segmentation ด้วย K-means
- แบ่งกลุ่มลูกค้าออกเป็น 4 กลุ่ม
- กราฟแสดงการกระจายกลุ่ม
- วิเคราะห์พฤติกรรมการใช้จ่าย

### 📊 Business Intelligence
- Dashboard แสดงข้อมูลทางธุรกิจ
- รายได้รวม, ยอดเฉลี่ย, จำนวนลูกค้า
- เทรนด์ยอดขายรายวัน
- ยอดขายตามช่วงเวลา
- Top Performers (เมนู, ลูกค้า, หมวดหมู่)

### 📈 Model Performance
- ประเมินประสิทธิภาพโมเดล AI
- Metrics: Precision, Recall, NDCG, MRR, Diversity, Coverage
- กราฟเปรียบเทียบ Metrics
- ข้อมูลโมเดลและ Matrix

---

## การใช้งาน

### การทดสอบระบบแนะนำ:
1. เลือกหน้า "🤖 ทดสอบระบบแนะนำ"
2. เลือกลูกค้าจาก dropdown
3. ปรับจำนวนเมนูที่ต้องการแนะนำ
4. กดปุ่ม "🎯 สร้างคำแนะนำ"
5. ดูผลลัพธ์และเปรียบเทียบกับประวัติ

### การดู Business Intelligence:
1. เลือกหน้า "📊 Business Intelligence"
2. ดูข้อมูลสถิติและกราฟต่างๆ
3. วิเคราะห์เทรนด์และ Top Performers

### การประเมินโมเดล:
1. เลือกหน้า "📈 Model Performance"
2. รอให้ระบบประเมินโมเดล
3. ดูผลลัพธ์ Metrics ต่างๆ

---

## คุณสมบัติพิเศษ

### 🎨 UI/UX Features:
- Responsive Design
- Interactive Charts (Plotly)
- Beautiful Cards และ Metrics
- Color-coded Sections
- Loading Spinners

### ⚡ Performance Features:
- Data Caching (`@st.cache_data`)
- Model Caching (`@st.cache_resource`)
- Efficient Data Loading
- Error Handling

### 🔧 Technical Features:
- Module Import System
- Fallback Mechanisms  
- Exception Handling
- Real-time Recommendations

---

## การแก้ไขปัญหา

### ถ้า Import Error:
- ตรวจสอบ path ใน `sys.path.append()`
- ตรวจสอบว่าไฟล์ module อยู่ใน `src/` folder

### ถ้าโมเดลไม่ทำงาน:
- ตรวจสอบ user_item_matrix
- ตรวจสอบ rating column ใน df_orders

### ถ้าข้อมูลไม่แสดง:
- ตรวจสอบ DataGenerator
- ตรวจสอบ data path

---

## การปรับแต่งเพิ่มเติม

### เพิ่มหน้าใหม่:
1. สร้างฟังก์ชัน `show_new_page()`
2. เพิ่มใน `selectbox` options
3. เพิ่ม condition ใน main()

### เพิ่ม Metrics:
1. เพิ่มใน `show_model_performance()`
2. อัพเดต evaluation results
3. เพิ่มกราฟแสดงผล

### ปรับแต่ง Styling:
1. แก้ไข CSS ใน `st.markdown()`
2. เพิ่ม custom classes
3. ปรับสี themes
