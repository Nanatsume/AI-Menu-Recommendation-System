"""
🍽️ AI Menu Recommendation System - Web Dashboard
เว็บแอปพลิเคชันสำหรับทดสอบระบบแนะนำเมนูอาหารอัจฉริยะ
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# ปิด TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# เพิ่ม path สำหรับ import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)
src_dir = os.path.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

# Import modules
try:
    from src.data_generation import DataGenerator
    from src.model_factory import create_simple_matrix_factorization, ModelFactory
    from src.evaluation_fixed import AdvancedRecommendationEvaluator
    from src.advanced_features import CustomerSegmentation, TrendAnalyzer, ContentBasedRecommender
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.stop()

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="🍽️ AI Menu Recommendation System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS สำหรับ styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: rgba(30, 35, 42, 0.8);
        color: #fafafa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    .recommendation-card h4 {
        color: #4ECDC4 !important;
    }
    .recommendation-card p {
        color: #e0e0e0 !important;
        margin: 0.3rem 0 !important;
    }
    .recommendation-card strong {
        color: #fafafa !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """โหลดข้อมูลและเก็บใน cache"""
    try:
        # ตรวจสอบว่ามีไฟล์ข้อมูลอยู่แล้วหรือไม่
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        customers_file = os.path.join(data_dir, 'customers.csv')
        menu_file = os.path.join(data_dir, 'menu.csv')
        orders_file = os.path.join(data_dir, 'orders.csv')
        
        # ถ้ามีไฟล์อยู่แล้ว ให้โหลดจากไฟล์
        if os.path.exists(customers_file) and os.path.exists(menu_file) and os.path.exists(orders_file):
            st.info("📂 โหลดข้อมูลจากไฟล์ที่มีอยู่แล้ว...")
            df_customers = pd.read_csv(customers_file)
            df_menu = pd.read_csv(menu_file)
            df_orders = pd.read_csv(orders_file)
        else:
            # ถ้าไม่มีไฟล์ ให้สร้างใหม่
            st.info("🔄 สร้างข้อมูลใหม่...")
            generator = DataGenerator()
            df_customers, df_menu, df_orders = generator.generate_all_data()
        
        # ตรวจสอบและแก้ไขโครงสร้างข้อมูล
        # แก้ไข column names ถ้าจำเป็น
        if 'menu_name' not in df_orders.columns and 'name' in df_menu.columns:
            # Merge เพื่อได้ menu_name
            df_orders = df_orders.merge(df_menu[['menu_id', 'name', 'category']], on='menu_id', how='left')
            df_orders = df_orders.rename(columns={'name': 'menu_name'})
        
        # เพิ่ม quantity column ถ้าไม่มี
        if 'quantity' not in df_orders.columns:
            df_orders['quantity'] = 1  # ตั้งค่าเริ่มต้นเป็น 1
        
        return df_customers, df_menu, df_orders
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

# Cache model training
@st.cache_resource
def load_model(user_item_matrix):
    """สร้างและฝึกโมเดลแล้วเก็บใน cache"""
    try:
        model = create_simple_matrix_factorization(user_item_matrix, n_components=30)
        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

@st.cache_data
def create_realistic_ratings(df_orders, df_menu, df_customers):
    """สร้าง rating ที่สมเหตุสมผลจากข้อมูลจริง"""
    df_orders = df_orders.copy()
    
    # ใช้ seed เพื่อให้ผลลัพธ์คงที่
    np.random.seed(42)
    
    # คำนวณ rating จากปัจจัยต่างๆ
    ratings = []
    
    for _, order in df_orders.iterrows():
        # Base rating (3.0-5.0)
        base_rating = 3.5
        
        # ปรับจาก popularity ของเมนู
        if 'menu_id' in order and order['menu_id'] in df_menu['menu_id'].values:
            menu_info = df_menu[df_menu['menu_id'] == order['menu_id']].iloc[0]
            if 'popularity' in menu_info:
                popularity_factor = (menu_info['popularity'] - 3.0) * 0.3  # ปรับ scale
                base_rating += popularity_factor
        
        # ปรับจาก budget ของลูกค้า vs ราคาเมนู
        if 'customer_id' in order and order['customer_id'] in df_customers['customer_id'].values:
            customer_info = df_customers[df_customers['customer_id'] == order['customer_id']].iloc[0]
            if 'avg_budget' in customer_info and 'amount' in order:
                # ถ้าเมนูอยู่ในงบประมาณ rating จะสูงกว่า
                budget_ratio = order['amount'] / customer_info['avg_budget']
                if budget_ratio <= 0.5:  # ถูกกว่างบ
                    base_rating += 0.3
                elif budget_ratio > 1.2:  # แพงเกินงบ
                    base_rating -= 0.4
        
        # ปรับจาก quantity (สั่งเยอะ = ชอบมาก)
        if 'quantity' in order and order['quantity'] > 1:
            quantity_bonus = min(0.4, (order['quantity'] - 1) * 0.1)
            base_rating += quantity_bonus
        
        # เพิ่มความหลากหลายโดยใช้ customer_id เป็น seed
        customer_variance = np.random.normal(0, 0.2)  # variation เล็กน้อย
        base_rating += customer_variance
        
        # จำกัดไว้ระหว่าง 1.0 - 5.0
        final_rating = max(1.0, min(5.0, base_rating))
        ratings.append(round(final_rating, 1))
    
    df_orders['rating'] = ratings
    return df_orders

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ AI Menu Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### ระบบแนะนำเมนูอาหารอัจฉริยะ - Interactive Dashboard")
    
    # โหลดข้อมูล
    with st.spinner("🔄 กำลังโหลดข้อมูล..."):
        df_customers, df_menu, df_orders = load_data()
    
    if df_customers is None:
        st.error("❌ ไม่สามารถโหลดข้อมูลได้")
        return
    
    # สร้าง user-item matrix
    try:
        # เพิ่ม rating column ถ้าไม่มี โดยใช้ข้อมูลจริงแทนการสุ่ม
        if 'rating' not in df_orders.columns:
            # สร้าง rating จากข้อมูลจริง
            df_orders = create_realistic_ratings(df_orders, df_menu, df_customers)
        
        user_item_matrix = df_orders.pivot_table(
            index='customer_id', 
            columns='menu_id', 
            values='rating',
            fill_value=0
        )
    except Exception as e:
        st.error(f"Error creating user-item matrix: {e}")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ การตั้งค่า")
    
    # เลือกหน้า
    page = st.sidebar.selectbox(
        "เลือกหน้าที่ต้องการ",
        ["🏠 หน้าหลัก", "🤖 ทดสอบระบบแนะนำ", "👥 วิเคราะห์ลูกค้า", "📊 Business Intelligence", "📈 Model Performance"]
    )
    
    st.sidebar.markdown("---")
    
    # ปุ่มรีเซ็ตข้อมูล
    if st.sidebar.button("🔄 สร้างข้อมูลใหม่", help="สร้างข้อมูลจำลองใหม่ทั้งหมด"):
        # ลบ cache และรีโหลดหน้า
        st.cache_data.clear()
        st.cache_resource.clear()
        # ลบไฟล์ข้อมูลเก่า
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        for filename in ['customers.csv', 'menu.csv', 'orders.csv']:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        st.rerun()
    
    # ปุ่มล้าง cache
    if st.sidebar.button("🧹 ล้าง Cache", help="ล้าง cache ทั้งหมดเพื่อรีเฟรชการวิเคราะห์"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # แสดงสถิติพื้นฐาน
    st.sidebar.header("📊 สถิติระบบ")
    st.sidebar.metric("👥 จำนวนลูกค้า", len(df_customers))
    st.sidebar.metric("🍽️ จำนวนเมนู", len(df_menu))
    st.sidebar.metric("📝 จำนวนออเดอร์", len(df_orders))
    st.sidebar.metric("💰 ยอดขายรวม", f"{df_orders['amount'].sum():,.0f} บาท")
    
    # แสดงหน้าตามที่เลือก
    if page == "🏠 หน้าหลัก":
        show_homepage(df_customers, df_menu, df_orders)
    elif page == "🤖 ทดสอบระบบแนะนำ":
        show_recommendation_page(df_customers, df_menu, df_orders, user_item_matrix)
    elif page == "👥 วิเคราะห์ลูกค้า":
        show_customer_analysis(df_customers, df_menu, df_orders)
    elif page == "📊 Business Intelligence":
        show_business_intelligence(df_customers, df_menu, df_orders)
    elif page == "📈 Model Performance":
        show_model_performance(df_customers, df_menu, df_orders, user_item_matrix)

def show_homepage(df_customers, df_menu, df_orders):
    """หน้าหลัก - แสดงภาพรวมระบบ"""
    
    st.header("🎯 ภาพรวมระบบแนะนำเมนู")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-container">
            <h3>👥 {len(df_customers)}</h3>
            <p>ลูกค้าทั้งหมด</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-container">
            <h3>🍽️ {len(df_menu)}</h3>
            <p>เมนูทั้งหมด</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-container">
            <h3>📝 {len(df_orders):,}</h3>
            <p>ออเดอร์ทั้งหมด</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        avg_order = df_orders['amount'].mean()
        st.markdown(f'''
        <div class="metric-container">
            <h3>💰 {avg_order:.0f}</h3>
            <p>ยอดเฉลี่ย/ออเดอร์</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 ยอดขายตามหมวดหมู่")
        try:
            # ใช้ column ที่มีอยู่จริง
            if 'category' in df_orders.columns:
                category_sales = df_orders.groupby('category')['amount'].sum().reset_index()
            else:
                # merge กับ df_menu เพื่อได้ category
                merged_data = df_orders.merge(df_menu, on='menu_id', how='left')
                category_sales = merged_data.groupby('category')['amount'].sum().reset_index()
            
            fig_pie = px.pie(category_sales, values='amount', names='category', 
                            title="การกระจายยอดขายตามหมวดหมู่")
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating category chart: {e}")
    
    with col2:
        st.subheader("🏆 เมนูยอดนิยม Top 10")
        try:
            # ใช้ column ที่มีอยู่จริง
            if 'menu_name' in df_orders.columns and 'quantity' in df_orders.columns:
                popular_menus = df_orders.groupby('menu_name')['quantity'].sum().nlargest(10).reset_index()
            elif 'menu_name' in df_orders.columns and 'amount' in df_orders.columns:
                popular_menus = df_orders.groupby('menu_name')['amount'].sum().nlargest(10).reset_index()
                popular_menus.columns = ['menu_name', 'quantity']
            else:
                # ใช้ข้อมูลจาก merge
                merged_data = df_orders.merge(df_menu, on='menu_id', how='left')
                if 'menu_name_y' in merged_data.columns:
                    popular_menus = merged_data.groupby('menu_name_y')['amount'].sum().nlargest(10).reset_index()
                    popular_menus.columns = ['menu_name', 'quantity']
                else:
                    popular_menus = merged_data.groupby('menu_name')['amount'].sum().nlargest(10).reset_index()
                    popular_menus.columns = ['menu_name', 'quantity']
                    
            fig_bar = px.bar(popular_menus, x='quantity', y='menu_name', 
                            orientation='h', title="เมนูที่ขายดีที่สุด")
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating popular menu chart: {e}")
    
    # Recent Activity
    st.subheader("📈 กิจกรรมล่าสุด")
    try:
        # แปลง order_date เป็น datetime ถ้ายังเป็น string
        df_orders_copy = df_orders.copy()
        if df_orders_copy['order_date'].dtype == 'object':
            df_orders_copy['order_date'] = pd.to_datetime(df_orders_copy['order_date'])
        
        recent_orders = df_orders_copy.nlargest(10, 'order_date')[['customer_id', 'menu_name', 'quantity', 'amount', 'order_date']]
        st.dataframe(recent_orders, use_container_width=True)
    except Exception as e:
        st.error(f"Error showing recent activity: {e}")
        # แสดงข้อมูลแบบง่ายๆ แทน
        df_orders_copy = df_orders.copy()
        recent_orders = df_orders_copy.tail(10)[['customer_id', 'menu_name', 'quantity', 'amount', 'order_date']]
        st.dataframe(recent_orders, use_container_width=True)

def show_recommendation_page(df_customers, df_menu, df_orders, user_item_matrix):
    """หน้าทดสอบระบบแนะนำ"""
    
    st.header("🤖 ทดสอบระบบแนะนำเมนู")
    
    # โหลดโมเดล
    with st.spinner("🔄 กำลังฝึกโมเดล AI..."):
        model = load_model(user_item_matrix)
    
    if model is None:
        st.error("❌ ไม่สามารถสร้างโมเดลได้")
        return
    
    st.success("✅ โมเดล AI พร้อมใช้งาน!")
    
    # เลือกลูกค้า
    st.subheader("👤 เลือกลูกค้าเพื่อทดสอบ")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # สร้าง customer display names
        customer_options = []
        for _, customer in df_customers.head(20).iterrows():  # แสดงแค่ 20 คนแรก
            display_name = f"{customer['customer_id']} (อายุ {customer['age']}, งบ {customer['avg_budget']:.0f}บ)"
            customer_options.append((customer['customer_id'], display_name))
        
        selected_customer_id = st.selectbox(
            "เลือกลูกค้า:",
            options=[cid for cid, _ in customer_options],
            format_func=lambda x: next(name for cid, name in customer_options if cid == x)
        )
        
        # จำนวนเมนูที่แนะนำ
        top_k = st.slider("จำนวนเมนูที่แนะนำ:", 5, 20, 10)
    
    with col2:
        # แสดงข้อมูลลูกค้า
        customer_info = df_customers[df_customers['customer_id'] == selected_customer_id].iloc[0]
        st.markdown(f"""
        **👤 ข้อมูลลูกค้า:**
        - **ID:** {customer_info['customer_id']}
        - **อายุ:** {customer_info['age']} ปี
        - **เพศ:** {'ชาย' if customer_info['gender'] == 'M' else 'หญิง'}
        - **งบประมาณเฉลี่ย:** {customer_info['avg_budget']:.0f} บาท
        - **เวลาที่ชอบ:** {customer_info['preferred_time']}
        """)
    
    if st.button("🎯 สร้างคำแนะนำ", type="primary"):
        try:
            # หา user index
            user_idx = list(user_item_matrix.index).index(selected_customer_id)
            
            # สร้างคำแนะนำ
            with st.spinner("🤖 กำลังสร้างคำแนะนำ..."):
                recommendations = model.predict_for_user(user_idx, top_k=top_k)
            
            st.subheader(f"🍽️ เมนูที่แนะนำสำหรับ {selected_customer_id}")
            
            # แสดงคำแนะนำในรูป cards
            cols = st.columns(2)
            
            for i, (menu_id, score) in enumerate(recommendations):
                try:
                    menu_info = df_menu[df_menu['menu_id'] == menu_id].iloc[0]
                    
                    with cols[i % 2]:
                        st.markdown(f'''
                        <div class="recommendation-card">
                            <h4>#{i+1} {menu_info['menu_name']}</h4>
                            <p><strong>หมวดหมู่:</strong> {menu_info['category']}</p>
                            <p><strong>ราคา:</strong> {menu_info['price']:.0f} บาท</p>
                            <p><strong>ความนิยม:</strong> {menu_info['popularity']:.1f}/5.0</p>
                            <p><strong>AI Score:</strong> {score:.3f}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying menu {menu_id}: {e}")
            
            # แสดงประวัติการสั่งของลูกค้า
            st.subheader("📝 ประวัติการสั่งอาหาร")
            customer_history = df_orders[df_orders['customer_id'] == selected_customer_id]
            
            if len(customer_history) > 0:
                history_summary = customer_history.groupby('menu_name').agg({
                    'quantity': 'sum',
                    'amount': 'sum',
                    'rating': 'mean'
                }).round(2).sort_values('quantity', ascending=False)
                
                st.dataframe(history_summary, use_container_width=True)
            else:
                st.info("ลูกค้าคนนี้ยังไม่เคยสั่งอาหาร")
                
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")

@st.cache_data
def cached_customer_segmentation(df_orders, df_customers, df_menu):
    """แบ่งกลุ่มลูกค้าและเก็บผลใน cache"""
    try:
        segmentation = CustomerSegmentation(n_clusters=4)
        customer_features = segmentation.create_customer_features(df_orders, df_customers, df_menu)
        cluster_labels, segment_names = segmentation.fit_segments(customer_features)
        return cluster_labels, segment_names, customer_features
    except Exception as e:
        st.error(f"Error in customer segmentation: {e}")
        return None, None, None

def show_customer_analysis(df_customers, df_menu, df_orders):
    """หน้าวิเคราะห์ลูกค้า"""
    
    st.header("👥 การวิเคราะห์ลูกค้า")
    
    # อธิบายเกี่ยวกับ K-Means Clustering
    with st.expander("ℹ️ เกี่ยวกับการแบ่งกลุ่มลูกค้า (K-Means Clustering)"):
        st.write("""
        **🤖 K-Means Clustering คืออะไร?**
        - เป็นอัลกอริทึม Machine Learning ที่แบ่งลูกค้าออกเป็น 4 กลุ่มตามพฤติกรรม
        - ใช้ข้อมูล: อายุ, งบประมาณ, ความถี่การสั่ง, rating, ความหลากหลายเมนู
        
        **🎯 ประโยชน์:**
        - วางแผนการตลาดเฉพาะกลุ่ม
        - ปรับกลยุทธ์ให้เหมาะสมกับแต่ละกลุ่ม
        - คาดการณ์พฤติกรรมลูกค้า
        """)
    
    # Customer Segmentation
    try:
        with st.spinner("🔄 กำลังแบ่งกลุ่มลูกค้า..."):
            cluster_labels, segment_names, customer_features = cached_customer_segmentation(df_orders, df_customers, df_menu)
        
        if cluster_labels is None:
            st.error("❌ ไม่สามารถแบ่งกลุ่มลูกค้าได้")
            return
        
        st.subheader("🏷️ การแบ่งกลุ่มลูกค้า (K-Means Clustering)")
        
        # แสดงข้อมูลการแบ่งกลุ่ม
        segment_counts = pd.Series(cluster_labels).value_counts()
        segment_df = pd.DataFrame({
            'Cluster': segment_counts.index,
            'Count': segment_counts.values,
            'Percentage': (segment_counts.values / len(cluster_labels) * 100).round(1),
            'Segment_Name': [segment_names[i] for i in segment_counts.index]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(segment_df, values='Count', names='Segment_Name', 
                           title="การกระจายกลุ่มลูกค้า")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📊 รายละเอียดแต่ละกลุ่ม")
            
            # แสดงข้อมูลแต่ละกลุ่มแบบ expandable
            for _, row in segment_df.iterrows():
                cluster_id = row['Cluster']
                
                # แก้ไข boolean indexing issue
                cluster_mask = np.array(cluster_labels) == cluster_id
                
                # ตรวจสอบขนาด mask และ DataFrame
                if len(cluster_mask) == len(customer_features):
                    cluster_customers = customer_features[cluster_mask]
                else:
                    # ถ้าขนาดไม่ตรง ใช้ iloc แทน
                    cluster_indices = np.where(np.array(cluster_labels) == cluster_id)[0]
                    cluster_customers = customer_features.iloc[cluster_indices]
                
                with st.expander(f"{row['Segment_Name']} - {row['Count']} คน ({row['Percentage']:.1f}%)"):
                    if len(cluster_customers) > 0:
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            avg_freq = cluster_customers['order_frequency'].mean()
                            st.metric("ความถี่สั่งเฉลี่ย", f"{avg_freq:.1f} ครั้ง")
                        
                        with col_b:
                            avg_budget = cluster_customers['avg_budget'].mean()
                            st.metric("งบประมาณเฉลี่ย", f"฿{avg_budget:,.0f}")
                        
                        with col_c:
                            avg_rating = cluster_customers['avg_rating'].mean()
                            st.metric("Rating เฉลี่ย", f"{avg_rating:.2f}/5.0")
                        
                        # แสดงข้อมูลเพิ่มเติม
                        st.write("**📈 ลักษณะเด่นของกลุ่ม:**")
                        col_x, col_y = st.columns(2)
                        
                        with col_x:
                            if 'recency' in cluster_customers.columns:
                                recency_avg = cluster_customers['recency'].mean()
                                st.write(f"• ระยะห่างการสั่งเฉลี่ย: {recency_avg:.1f} วัน")
                            
                            if 'category_diversity' in cluster_customers.columns:
                                cat_div = cluster_customers['category_diversity'].mean()
                                st.write(f"• ความหลากหลายหมวดหมู่: {cat_div:.1f}")
                        
                        with col_y:
                            if 'total_quantity' in cluster_customers.columns:
                                total_qty = cluster_customers['total_quantity'].mean()
                                st.write(f"• จำนวนรายการเฉลี่ย: {total_qty:.1f}")
                            
                            if 'menu_diversity' in cluster_customers.columns:
                                menu_div = cluster_customers['menu_diversity'].mean()
                                st.write(f"• ความหลากหลายเมนู: {menu_div:.1f}")
                    else:
                        st.error(f"ไม่พบข้อมูลสำหรับกลุ่ม {cluster_id}")
        
        # Customer behavior analysis
        st.subheader("📈 พฤติกรรมลูกค้า")
        
        # สร้างกราฟ scatter plot
        customer_stats = df_orders.groupby('customer_id').agg({
            'rating': ['count', 'mean'],
            'amount': 'sum'
        }).round(2)
        
        customer_stats.columns = ['order_frequency', 'avg_rating', 'total_spent']
        customer_stats = customer_stats.reset_index()
        
        fig_scatter = px.scatter(
            customer_stats, 
            x='order_frequency', 
            y='total_spent',
            size='avg_rating',
            hover_data=['customer_id'],
            title="ความสัมพันธ์: ความถี่การสั่ง vs ยอดใช้จ่าย",
            labels={
                'order_frequency': 'ความถี่การสั่งอาหาร (ครั้ง)',
                'total_spent': 'ยอดใช้จ่ายรวม (บาท)'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in customer analysis: {e}")

def show_business_intelligence(df_customers, df_menu, df_orders):
    """หน้า Business Intelligence"""
    
    st.header("📊 Business Intelligence Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = df_orders['amount'].sum()
    avg_order_value = df_orders['amount'].mean()
    total_customers = df_orders['customer_id'].nunique()
    total_orders = len(df_orders)
    
    with col1:
        st.metric("💰 รายได้รวม", f"{total_revenue:,.0f} บาท")
    with col2:
        st.metric("🛒 ยอดเฉลี่ย/ออเดอร์", f"{avg_order_value:.0f} บาท")
    with col3:
        st.metric("👥 ลูกค้าที่ซื้อ", f"{total_customers} คน")
    with col4:
        st.metric("📝 ออเดอร์ทั้งหมด", f"{total_orders:,} รายการ")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 เทรนด์ยอดขายรายวัน")
        daily_sales = df_orders.groupby('order_date')['amount'].sum().reset_index()
        daily_sales['order_date'] = pd.to_datetime(daily_sales['order_date'])
        
        fig_line = px.line(daily_sales, x='order_date', y='amount', 
                          title="ยอดขายรายวัน")
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col2:
        st.subheader("⏰ ยอดขายตามช่วงเวลา")
        
        # สร้างช่วงเวลา
        if 'order_time' in df_orders.columns:
            df_orders['hour'] = pd.to_datetime(df_orders['order_time'], format='%H:%M').dt.hour
        else:
            df_orders['hour'] = np.random.randint(7, 23, len(df_orders))
        
        hourly_sales = df_orders.groupby('hour')['amount'].sum().reset_index()
        
        fig_bar = px.bar(hourly_sales, x='hour', y='amount', 
                        title="ยอดขายตามชั่วโมง")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Top performers
    st.subheader("🏆 Top Performers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🍽️ เมนูขายดี**")
        try:
            if 'menu_name' in df_orders.columns and 'quantity' in df_orders.columns:
                top_menus = df_orders.groupby('menu_name').agg({
                    'quantity': 'sum',
                    'amount': 'sum'
                }).sort_values('quantity', ascending=False).head(5)
            else:
                # ใช้ amount แทน quantity
                top_menus = df_orders.groupby('menu_name')['amount'].sum().nlargest(5).reset_index()
                top_menus.columns = ['menu_name', 'amount']
            st.dataframe(top_menus)
        except Exception as e:
            st.error(f"Error in top menus: {e}")
    
    with col2:
        st.write("**👑 ลูกค้า VIP**")
        try:
            top_customers = df_orders.groupby('customer_id')['amount'].sum().nlargest(5).reset_index()
            st.dataframe(top_customers)
        except Exception as e:
            st.error(f"Error in top customers: {e}")
    
    with col3:
        st.write("**📊 หมวดหมู่ยอดนิยม**")
        try:
            # ตรวจสอบว่า df_orders มี category อยู่แล้วหรือไม่
            if 'category' in df_orders.columns:
                # ใช้ category จาก df_orders โดยตรง
                if 'quantity' in df_orders.columns:
                    category_performance = df_orders.groupby('category').agg({
                        'amount': 'sum',
                        'quantity': 'sum'
                    }).sort_values('amount', ascending=False)
                else:
                    category_performance = df_orders.groupby('category')['amount'].sum().reset_index()
                    category_performance.columns = ['category', 'amount']
                st.dataframe(category_performance)
            else:
                # ถ้าไม่มี category ใน df_orders ให้ merge กับ df_menu
                merged_data = None
                if 'menu_id' in df_orders.columns and 'menu_id' in df_menu.columns:
                    merged_data = df_orders.merge(df_menu, on='menu_id', how='left')
                elif 'menu_name' in df_orders.columns and 'menu_name' in df_menu.columns:
                    merged_data = df_orders.merge(df_menu, on='menu_name', how='left')
                
                if merged_data is not None and 'category' in merged_data.columns:
                    if 'quantity' in merged_data.columns:
                        category_performance = merged_data.groupby('category').agg({
                            'amount': 'sum',
                            'quantity': 'sum'
                        }).sort_values('amount', ascending=False)
                    else:
                        category_performance = merged_data.groupby('category')['amount'].sum().reset_index()
                        category_performance.columns = ['category', 'amount']
                    st.dataframe(category_performance)
                else:
                    # ถ้าไม่สามารถ merge ได้ แสดงข้อมูลทั่วไปจาก df_menu
                    st.info("แสดงข้อมูลหมวดหมู่ทั่วไปจากเมนู")
                    if 'category' in df_menu.columns:
                        category_counts = df_menu['category'].value_counts().reset_index()
                        category_counts.columns = ['category', 'จำนวนเมนู']
                        st.dataframe(category_counts)
                    else:
                        st.error("ไม่พบข้อมูลหมวดหมู่")
                        
        except Exception as e:
            st.error(f"Error in category performance: {e}")
            # แสดงข้อมูลเมนูทั่วไปแทน
            try:
                if 'category' in df_menu.columns:
                    category_counts = df_menu['category'].value_counts().reset_index()
                    category_counts.columns = ['category', 'จำนวนเมนู']
                    st.dataframe(category_counts)
                else:
                    st.error("ไม่พบข้อมูลหมวดหมู่ในเมนู")
            except Exception as inner_e:
                st.error(f"ไม่สามารถแสดงข้อมูลหมวดหมู่ได้: {inner_e}")

def show_model_performance(df_customers, df_menu, df_orders, user_item_matrix):
    """หน้าประสิทธิภาพโมเดล"""
    
    st.header("📈 ประสิทธิภาพโมเดล AI")
    
    # โหลดโมเดล
    with st.spinner("🔄 กำลังฝึกและประเมินโมเดล..."):
        model = load_model(user_item_matrix)
    
    if model is None:
        st.error("❌ ไม่สามารถสร้างโมเดลได้")
        return
    
    try:
        # ประเมินโมเดล
        evaluator = AdvancedRecommendationEvaluator(model)
        results = evaluator.evaluate_comprehensive(user_item_matrix, df_menu, df_orders, test_ratio=0.2)
        
        if results:
            st.success("✅ การประเมินโมเดลเสร็จสิ้น!")
            
            # แสดงผลลัพธ์
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Precision@10", f"{results.get('precision@10', 0):.3f}")
                st.metric("📊 Recall@10", f"{results.get('recall@10', 0):.3f}")
            
            with col2:
                st.metric("⭐ NDCG@10", f"{results.get('ndcg@10', 0):.3f}")
                st.metric("🔄 MRR", f"{results.get('mrr', 0):.3f}")
            
            with col3:
                st.metric("🌈 Diversity", f"{results.get('diversity', 0):.3f}")
                st.metric("📺 Coverage", f"{results.get('catalog_coverage', 0):.3f}")
            
            # สร้างกราฟเปรียบเทียบ
            st.subheader("📊 เปรียบเทียบ Metrics")
            
            metrics_data = {
                'Metric': ['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10'],
                'Score': [
                    results.get('precision@5', 0),
                    results.get('precision@10', 0),
                    results.get('recall@5', 0),
                    results.get('recall@10', 0),
                    results.get('ndcg@5', 0),
                    results.get('ndcg@10', 0)
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            fig_bar = px.bar(metrics_df, x='Metric', y='Score', 
                           title="ประสิทธิภาพโมเดลตาม Metrics ต่างๆ")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # แสดงรายละเอียด
            st.subheader("📋 รายละเอียดผลการประเมิน")
            results_df = pd.DataFrame([results]).T
            results_df.columns = ['Score']
            results_df['Score'] = results_df['Score'].round(4)
            st.dataframe(results_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error evaluating model: {e}")
    
    # Model Information
    st.subheader("ℹ️ ข้อมูลโมเดล")
    st.write(f"""
    **🤖 ประเภทโมเดล:** Matrix Factorization
    **🔢 จำนวน Components:** 30
    **📊 ขนาด User-Item Matrix:** {user_item_matrix.shape[0]} × {user_item_matrix.shape[1]}
    **📈 Sparsity:** {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.2f}%
    """)

if __name__ == "__main__":
    main()
