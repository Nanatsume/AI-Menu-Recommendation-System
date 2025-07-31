"""
Streamlit Dashboard for AI Menu Recommendation System
แดชบอร์ดสำหรับระบบแนะนำเมนูอาหาร
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import HybridRecommendationSystem
from evaluation import RecommendationEvaluator

# กำหนดค่า Streamlit
st.set_page_config(
    page_title="🍽️ AI Menu Recommendation System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """โหลดข้อมูลและ cache ไว้"""
    try:
        customers = pd.read_csv('data/customers.csv')
        menu = pd.read_csv('data/menu.csv')
        orders = pd.read_csv('data/orders.csv')
        return customers, menu, orders
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์ข้อมูล กรุณารันไฟล์ data_generation.py ก่อน")
        return None, None, None

@st.cache_resource
def load_recommendation_system():
    """โหลดระบบแนะนำและ cache ไว้"""
    try:
        system = HybridRecommendationSystem()
        return system
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดระบบแนะนำได้: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ AI Menu Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # โหลดข้อมูล
    customers, menu, orders = load_data()
    if customers is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ เมนูควบคุม")
    
    # เลือกหน้า
    page = st.sidebar.selectbox(
        "เลือกหน้า",
        ["🏠 หน้าหลัก", "👤 คำแนะนำส่วนบุคคล", "📊 วิเคราะห์ข้อมูล", "📈 ประเมินผลโมเดล"]
    )
    
    if page == "🏠 หน้าหลัก":
        show_home_page(customers, menu, orders)
    elif page == "👤 คำแนะนำส่วนบุคคล":
        show_recommendation_page(customers, menu, orders)
    elif page == "📊 วิเคราะห์ข้อมูล":
        show_analytics_page(customers, menu, orders)
    elif page == "📈 ประเมินผลโมเดล":
        show_evaluation_page()

def show_home_page(customers, menu, orders):
    """หน้าหลัก - ภาพรวมของระบบ"""
    st.header("🏠 ภาพรวมระบบ")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 จำนวนลูกค้า",
            value=f"{len(customers):,}",
            delta="Active Users"
        )
    
    with col2:
        st.metric(
            label="🍽️ จำนวนเมนู",
            value=f"{len(menu):,}",
            delta="Available Items"
        )
    
    with col3:
        st.metric(
            label="📝 จำนวนออเดอร์",
            value=f"{len(orders):,}",
            delta="Total Orders"
        )
    
    with col4:
        total_revenue = orders['amount'].sum()
        st.metric(
            label="💰 ยอดขายรวม",
            value=f"{total_revenue:,.2f} ฿",
            delta="Revenue"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 การกระจายตัวของหมวดหมู่เมนู")
        category_counts = menu['category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="สัดส่วนเมนูแต่ละหมวดหมู่"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("👥 การกระจายตัวของอายุลูกค้า")
        age_bins = pd.cut(customers['age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        age_counts = age_bins.value_counts()
        fig = px.bar(
            x=age_counts.index,
            y=age_counts.values,
            title="จำนวนลูกค้าแต่ละช่วงอายุ"
        )
        fig.update_layout(xaxis_title="ช่วงอายุ", yaxis_title="จำนวนคน")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sales Analysis
    st.subheader("📈 การวิเคราะห์ยอดขาย")
    
    # รายวัน
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    daily_sales = orders.groupby('order_date').agg({
        'amount': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=daily_sales['order_date'], y=daily_sales['amount'], name="ยอดขาย (฿)"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=daily_sales['order_date'], y=daily_sales['order_id'], name="จำนวนออเดอร์"),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="วันที่")
    fig.update_yaxes(title_text="ยอดขาย (บาท)", secondary_y=False)
    fig.update_yaxes(title_text="จำนวนออเดอร์", secondary_y=True)
    fig.update_layout(title_text="แนวโน้มยอดขายรายวัน")
    
    st.plotly_chart(fig, use_container_width=True)

def show_recommendation_page(customers, menu, orders):
    """หน้าคำแนะนำส่วนบุคคล"""
    st.header("👤 คำแนะนำเมนูส่วนบุคคล")
    
    # เลือกลูกค้า
    customer_ids = customers['customer_id'].tolist()
    selected_customer = st.selectbox("เลือกลูกค้า:", customer_ids)
    
    if selected_customer:
        # แสดงข้อมูลลูกค้า
        customer_info = customers[customers['customer_id'] == selected_customer].iloc[0]
        customer_orders = orders[orders['customer_id'] == selected_customer]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 ข้อมูลลูกค้า")
            
            st.markdown(f"""
            <div class="metric-card">
            <h4>👤 {selected_customer}</h4>
            <p><strong>🎂 อายุ:</strong> {customer_info['age']} ปี</p>
            <p><strong>⚧ เพศ:</strong> {'ชาย' if customer_info['gender'] == 'M' else 'หญิง'}</p>
            <p><strong>💰 งบประมาณเฉลี่ย:</strong> {customer_info['avg_budget']:.2f} บาท</p>
            <p><strong>⏰ เวลาที่ชอบ:</strong> {customer_info['preferred_time']}</p>
            <p><strong>📊 จำนวนออเดอร์:</strong> {len(customer_orders)} ครั้ง</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🍽️ ประวัติการสั่งอาหาร")
            if len(customer_orders) > 0:
                category_orders = customer_orders['category'].value_counts()
                fig = px.bar(
                    x=category_orders.values,
                    y=category_orders.index,
                    orientation='h',
                    title="จำนวนครั้งที่สั่งแต่ละหมวดหมู่"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ลูกค้าใหม่ - ยังไม่มีประวัติการสั่งอาหาร")
        
        # โหลดระบบแนะนำ
        st.markdown("---")
        st.subheader("🤖 คำแนะนำจาก AI")
        
        if st.button("🎯 ขอคำแนะนำเมนู", type="primary"):
            with st.spinner("กำลังวิเคราะห์และแนะนำเมนู..."):
                try:
                    # สำหรับ demo ใช้การแนะนำแบบง่าย
                    recommended_menus = get_simple_recommendations(customer_info, menu, customer_orders)
                    
                    st.success("✅ คำแนะนำเมนูพร้อมแล้ว!")
                    
                    col1, col2 = st.columns(2)
                    
                    for i, (idx, rec) in enumerate(recommended_menus.iterrows()):
                        col = col1 if i % 2 == 0 else col2
                        
                        with col:
                            st.markdown(f"""
                            <div class="recommendation-card">
                            <h4>🍽️ {rec['menu_name']}</h4>
                            <p><strong>📂 หมวดหมู่:</strong> {rec['category']}</p>
                            <p><strong>💰 ราคา:</strong> {rec['price']:.2f} บาท</p>
                            <p><strong>⭐ คะแนนความนิยม:</strong> {rec['popularity']:.1f}/5.0</p>
                            <p><strong>💡 เหตุผล:</strong> {rec['reason']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {e}")

def get_simple_recommendations(customer_info, menu, customer_orders, top_k=6):
    """ระบบแนะนำแบบง่าย สำหรับ demo"""
    
    # ลองิกการแนะนำแบบง่าย
    budget = customer_info['avg_budget']
    age = customer_info['age']
    
    # กรองเมนูตามงบประมาณ
    affordable_menu = menu[menu['price'] <= budget * 0.8].copy()
    
    if len(affordable_menu) == 0:
        affordable_menu = menu.copy()
    
    # หาหมวดหมู่ที่ชอบ
    if len(customer_orders) > 0:
        favorite_categories = customer_orders['category'].value_counts().index.tolist()
    else:
        favorite_categories = ['อาหารหลัก', 'เครื่องดื่ม', 'ของหวาน']
    
    recommendations = []
    reasons = []
    
    # แนะนำตามหมวดหมู่ที่ชอบและความนิยม
    for category in favorite_categories:
        cat_menus = affordable_menu[affordable_menu['category'] == category].nlargest(2, 'popularity')
        for _, menu_item in cat_menus.iterrows():
            if len(recommendations) < top_k:
                recommendations.append(menu_item)
                if len(customer_orders) > 0:
                    reasons.append(f"ตรงกับหมวดหมู่ที่คุณชอบ ({category})")
                else:
                    reasons.append(f"เมนูยอดนิยมในหมวด {category}")
    
    # เติมเมนูยอดนิยมเพิ่ม
    while len(recommendations) < top_k:
        remaining_menus = affordable_menu[~affordable_menu['menu_id'].isin([r['menu_id'] for r in recommendations])]
        if len(remaining_menus) == 0:
            break
        
        popular_menu = remaining_menus.nlargest(1, 'popularity').iloc[0]
        recommendations.append(popular_menu)
        reasons.append("เมนูยอดนิยมโดยรวม")
    
    # สร้าง DataFrame
    result_df = pd.DataFrame(recommendations[:top_k])
    result_df['reason'] = reasons[:top_k]
    
    return result_df

def show_analytics_page(customers, menu, orders):
    """หน้าวิเคราะห์ข้อมูล"""
    st.header("📊 การวิเคราะห์ข้อมูลเชิงลึก")
    
    # Analysis Options
    analysis_type = st.selectbox(
        "เลือกประเภทการวิเคราะห์:",
        ["📈 ยอดขายและแนวโน้ม", "👥 พฤติกรรมลูกค้า", "🍽️ ความนิยมของเมนู", "⏰ การวิเคราะห์เวลา"]
    )
    
    if analysis_type == "📈 ยอดขายและแนวโน้ม":
        show_sales_analysis(orders, menu)
    elif analysis_type == "👥 พฤติกรรมลูกค้า":
        show_customer_analysis(customers, orders)
    elif analysis_type == "🍽️ ความนิยมของเมนู":
        show_menu_analysis(menu, orders)
    elif analysis_type == "⏰ การวิเคราะห์เวลา":
        show_time_analysis(orders)

def show_sales_analysis(orders, menu):
    """วิเคราะห์ยอดขาย"""
    st.subheader("📈 การวิเคราะห์ยอดขาย")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top selling items
        top_items = orders.groupby('menu_name').agg({
            'quantity': 'sum',
            'amount': 'sum'
        }).sort_values('quantity', ascending=False).head(10)
        
        fig = px.bar(
            x=top_items['quantity'],
            y=top_items.index,
            orientation='h',
            title="🏆 Top 10 เมนูขายดี (จำนวน)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by category
        category_revenue = orders.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        fig = px.pie(
            values=category_revenue.values,
            names=category_revenue.index,
            title="💰 รายได้แยกตามหมวดหมู่"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_customer_analysis(customers, orders):
    """วิเคราะห์พฤติกรรมลูกค้า"""
    st.subheader("👥 การวิเคราะห์พฤติกรรมลูกค้า")
    
    # Customer segmentation
    customer_stats = orders.groupby('customer_id').agg({
        'amount': ['sum', 'mean', 'count']
    }).round(2)
    
    customer_stats.columns = ['total_spent', 'avg_order', 'order_count']
    customer_stats = customer_stats.merge(customers[['customer_id', 'age', 'gender']], 
                                        left_index=True, right_on='customer_id')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spending by age group
        age_bins = pd.cut(customer_stats['age'], bins=[0, 25, 35, 45, 55, 100], 
                         labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        age_spending = customer_stats.groupby(age_bins)['total_spent'].mean()
        
        fig = px.bar(
            x=age_spending.index,
            y=age_spending.values,
            title="💳 ค่าใช้จ่ายเฉลี่ยตามช่วงอายุ"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Order frequency distribution
        fig = px.histogram(
            customer_stats,
            x='order_count',
            title="📊 การกระจายตัวของความถี่การสั่งอาหาร"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_menu_analysis(menu, orders):
    """วิเคราะห์ความนิยมของเมนู"""
    st.subheader("🍽️ การวิเคราะห์ความนิยมของเมนู")
    
    menu_performance = orders.groupby('menu_id').agg({
        'quantity': 'sum',
        'amount': 'sum',
        'customer_id': 'nunique'
    }).rename(columns={'customer_id': 'unique_customers'})
    
    menu_performance = menu_performance.merge(menu, on='menu_id')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Popularity
        fig = px.scatter(
            menu_performance,
            x='price',
            y='quantity',
            size='unique_customers',
            color='category',
            title="💰 ราคา vs ความนิยม",
            hover_data=['menu_name']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category performance
        category_perf = menu_performance.groupby('category').agg({
            'quantity': 'sum',
            'amount': 'sum',
            'unique_customers': 'sum'
        })
        
        fig = px.bar(
            x=category_perf.index,
            y=category_perf['quantity'],
            title="📊 ประสิทธิภาพตามหมวดหมู่"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_time_analysis(orders):
    """วิเคราะห์ตามเวลา"""
    st.subheader("⏰ การวิเคราะห์ตามเวลา")
    
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders['day_of_week'] = orders['order_date'].dt.day_name()
    orders['hour'] = pd.to_datetime(orders['order_time']).dt.hour
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by day of week
        daily_sales = orders.groupby('day_of_week')['amount'].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sales = daily_sales.reindex(day_order)
        
        fig = px.bar(
            x=daily_sales.index,
            y=daily_sales.values,
            title="📅 ยอดขายแต่ละวันในสัปดาห์"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales by hour
        hourly_sales = orders.groupby('hour')['amount'].sum()
        
        fig = px.line(
            x=hourly_sales.index,
            y=hourly_sales.values,
            title="🕐 ยอดขายแต่ละชั่วโมง"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_evaluation_page():
    """หน้าประเมินผลโมเดล"""
    st.header("📈 การประเมินผลโมเดล")
    
    st.info("🚧 หน้านี้จะแสดงผลการประเมินโมเดล ML หลังจากที่ได้ฝึกโมเดลแล้ว")
    
    # Mock evaluation results for demo
    st.subheader("📊 ผลการประเมิน")
    
    metrics_data = {
        'Metric': ['Precision@5', 'Recall@5', 'F1-Score@5', 'Hit Rate@5', 'MRR@5'],
        'Score': [0.75, 0.68, 0.71, 0.85, 0.62]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            title="📈 คะแนนการประเมินโมเดล"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📝 สรุปผลการประเมิน")
    
    st.markdown("""
    ### ✅ จุดแข็งของโมเดล:
    - **Hit Rate สูง (85%)**: โมเดลสามารถแนะนำเมนูที่ลูกค้าสนใจได้ดี
    - **Precision ดี (75%)**: เมนูที่แนะนำส่วนใหญ่ตรงใจลูกค้า
    - **ระบบทำงานเสถียร**: ไม่มีข้อผิดพลาดในการประมวลผล
    
    ### ⚠️ จุดที่ควรปรับปรุง:
    - **Recall**: ยังแนะนำเมนูที่ลูกค้าชอบได้ไม่ครบ
    - **MRR**: ลำดับการแนะนำยังไม่เหมาะสมเท่าที่ควร
    - **ความหลากหลาย**: ควรเพิ่มความหลากหลายในการแนะนำ
    
    ### 🎯 ข้อเสนอแนะเพื่อพัฒนา:
    1. เพิ่มข้อมูล features เช่น รีวิว, เวลา, สภาพอากาศ
    2. ปรับปรุงอัลกอริทึม neural network
    3. เพิ่ม diversity ในการแนะนำ
    4. A/B testing กับลูกค้าจริง
    """)

if __name__ == "__main__":
    main()
