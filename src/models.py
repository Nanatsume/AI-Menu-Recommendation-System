"""
Machine Learning Models for AI Menu Recommendation System
โมเดลการแนะนำเมนูอาหาร
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle
import os

class MatrixFactorizationModel:
    """Matrix Factorization using SVD for Collaborative Filtering"""
    
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
        self.is_fitted = False
        
    def fit(self, user_item_matrix):
        """ฝึกโมเดล Matrix Factorization"""
        print(f"🧠 กำลังฝึกโมเดล Matrix Factorization (n_components={self.n_components})...")
        
        self.user_item_matrix = user_item_matrix
        self.user_factors = self.model.fit_transform(user_item_matrix)
        self.item_factors = self.model.components_.T
        
        # คำนวณ reconstructed matrix
        self.predicted_ratings = np.dot(self.user_factors, self.item_factors.T)
        
        self.is_fitted = True
        print("✅ ฝึกโมเดล Matrix Factorization เสร็จสิ้น")
        
    def predict_for_user(self, user_idx, top_k=10):
        """แนะนำเมนูสำหรับลูกค้าคนหนึ่ง"""
        if not self.is_fitted:
            raise ValueError("โมเดลยังไม่ได้ฝึก กรุณา fit ก่อน")
        
        # ดึงคะแนนทำนายสำหรับลูกค้าคนนี้
        user_predictions = self.predicted_ratings[user_idx]
        
        # หาเมนูที่ยังไม่เคยสั่ง (rating = 0)
        already_ordered = self.user_item_matrix.iloc[user_idx] > 0
        
        # ซ่อนเมนูที่เคยสั่งแล้ว
        user_predictions[already_ordered] = -np.inf
        
        # หา top-k เมนูที่แนะนำ
        top_indices = np.argsort(user_predictions)[::-1][:top_k]
        top_scores = user_predictions[top_indices]
        
        # แปลง indices เป็น menu_ids
        menu_columns = self.user_item_matrix.columns
        recommended_menu_ids = [menu_columns[i] for i in top_indices]
        
        # return เป็น list of tuples (menu_id, score)
        return list(zip(recommended_menu_ids, top_scores))
    
    def get_similar_users(self, user_idx, top_k=5):
        """หาลูกค้าที่มีความชอบคล้ายกัน"""
        user_similarities = cosine_similarity([self.user_factors[user_idx]], self.user_factors)[0]
        similar_indices = np.argsort(user_similarities)[::-1][1:top_k+1]  # ไม่เอาตัวเอง
        return similar_indices, user_similarities[similar_indices]

class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering Model"""
    
    def __init__(self, n_users, n_items, embedding_size=50, hidden_units=[128, 64]):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.model = None
        
    def build_model(self):
        """สร้างโมเดล Neural Collaborative Filtering"""
        print("🏗️ กำลังสร้างโมเดล Neural Collaborative Filtering...")
        
        # Input layers
        user_input = layers.Input(shape=(), name='user_id')
        item_input = layers.Input(shape=(), name='item_id')
        
        # Embedding layers
        user_embedding = layers.Embedding(self.n_users, self.embedding_size, name='user_embedding')(user_input)
        item_embedding = layers.Embedding(self.n_items, self.embedding_size, name='item_embedding')(item_input)
        
        # Flatten embeddings
        user_flat = layers.Flatten()(user_embedding)
        item_flat = layers.Flatten()(item_embedding)
        
        # Concatenate user and item embeddings
        concat = layers.Concatenate()([user_flat, item_flat])
        
        # Hidden layers
        x = concat
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='rating')(x)
        
        # Create model
        self.model = keras.Model(inputs=[user_input, item_input], outputs=output)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ สร้างโมเดล Neural Collaborative Filtering เสร็จสิ้น")
        return self.model
    
    def train(self, train_data, validation_data=None, epochs=50, batch_size=256):
        """ฝึกโมเดล"""
        if self.model is None:
            self.build_model()
        
        print("🚀 เริ่มฝึกโมเดล Neural Collaborative Filtering...")
        
        # เตรียมข้อมูล
        user_ids = train_data['user_idx'].values
        item_ids = train_data['item_idx'].values  
        ratings = train_data['rating'].values
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # ฝึกโมเดล
        history = self.model.fit(
            [user_ids, item_ids], ratings,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ ฝึกโมเดลเสร็จสิ้น")
        return history
    
    def predict_for_user(self, user_idx, item_indices, user_to_idx, item_to_idx):
        """ทำนายคะแนนสำหรับลูกค้าและเมนูที่กำหนด"""
        user_ids = np.full(len(item_indices), user_idx)
        predictions = self.model.predict([user_ids, item_indices], verbose=0)
        return predictions.flatten()
    
    def recommend_items(self, user_idx, all_items, already_ordered, top_k=10):
        """แนะนำเมนูสำหรับลูกค้า"""
        # หาเมนูที่ยังไม่เคยสั่ง
        candidate_items = [item for item in all_items if item not in already_ordered]
        
        if len(candidate_items) == 0:
            return [], []
        
        # ทำนายคะแนน
        predictions = self.predict_for_user(user_idx, candidate_items, None, None)
        
        # เรียงลำดับและเลือก top-k
        item_scores = list(zip(candidate_items, predictions))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_items = [item for item, _ in item_scores[:top_k]]
        top_scores = [score for _, score in item_scores[:top_k]]
        
        return top_items, top_scores

class HybridRecommendationSystem:
    """ระบบแนะนำแบบผสม (Hybrid) ที่รวม Matrix Factorization และ Neural CF"""
    
    def __init__(self, data_dir="data/processed"):
        self.data_dir = data_dir
        self.matrix_model = None
        self.neural_model = None
        self.load_mappings()
        
    def load_mappings(self):
        """โหลด mappings และข้อมูลที่จำเป็น"""
        print("📂 กำลังโหลด mappings และข้อมูล...")
        
        with open(f'{self.data_dir}/mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            self.user_to_idx = mappings['user_to_idx']
            self.item_to_idx = mappings['item_to_idx']
            self.idx_to_user = mappings['idx_to_user']
            self.idx_to_item = mappings['idx_to_item']
            self.user_ids = mappings['user_ids']
            self.item_ids = mappings['item_ids']
        
        # โหลดข้อมูลเพิ่มเติม
        self.user_features = pd.read_csv(f'{self.data_dir}/user_features.csv')
        self.item_features = pd.read_csv(f'{self.data_dir}/item_features.csv')
        
        print("✅ โหลดข้อมูลเสร็จสิ้น")
    
    def train_models(self):
        """ฝึกทั้งสองโมเดล"""
        print("🚀 เริ่มฝึกโมเดลทั้งหมด...")
        
        # โหลดข้อมูล
        train_data = pd.read_csv(f'{self.data_dir}/train_data.csv')
        user_item_matrix = pd.DataFrame(
            np.load(f'{self.data_dir}/user_item_matrix.npy'),
            index=self.user_ids,
            columns=self.item_ids
        )
        
        # 1. ฝึก Matrix Factorization
        self.matrix_model = MatrixFactorizationModel(n_components=50)
        self.matrix_model.fit(user_item_matrix)
        
        # 2. เตรียมข้อมูลสำหรับ Neural CF
        train_data['user_idx'] = train_data['customer_id'].map(self.user_to_idx)
        train_data['item_idx'] = train_data['menu_id'].map(self.item_to_idx)
        train_data = train_data.dropna()
        
        # 3. ฝึก Neural Collaborative Filtering
        self.neural_model = NeuralCollaborativeFiltering(
            n_users=len(self.user_to_idx),
            n_items=len(self.item_to_idx),
            embedding_size=50
        )
        
        self.neural_model.train(train_data, epochs=30)
        
        print("🎉 ฝึกโมเดลทั้งหมดเสร็จสิ้น!")
    
    def recommend_for_user(self, customer_id, top_k=10, weight_matrix=0.5, weight_neural=0.5):
        """แนะนำเมนูสำหรับลูกค้า โดยรวมผลจากทั้งสองโมเดล"""
        
        # ตรวจสอบว่าลูกค้าอยู่ในระบบหรือไม่
        if customer_id not in self.user_to_idx:
            return self._recommend_for_new_user(top_k)
        
        user_idx = self.user_to_idx[customer_id]
        
        # 1. ได้คำแนะนำจาก Matrix Factorization
        matrix_items, matrix_scores = self.matrix_model.predict_for_user(user_idx, top_k * 2)
        
        # 2. เตรียมข้อมูลสำหรับ Neural CF
        already_ordered = set()
        if customer_id in self.user_ids:
            # หาเมนูที่เคยสั่งแล้ว
            user_orders = pd.read_csv('data/orders.csv')
            user_orders = user_orders[user_orders['customer_id'] == customer_id]
            already_ordered = set(user_orders['menu_id'].values)
        
        # 3. ได้คำแนะนำจาก Neural CF
        all_items = list(range(len(self.item_to_idx)))
        neural_items, neural_scores = self.neural_model.recommend_items(
            user_idx, all_items, already_ordered, top_k * 2
        )
        
        # 4. รวมผลจากทั้งสองโมเดล
        combined_scores = {}
        
        # Matrix Factorization scores
        for item_idx, score in zip(matrix_items, matrix_scores):
            item_id = self.idx_to_item[item_idx]
            combined_scores[item_id] = weight_matrix * score
        
        # Neural CF scores
        for item_idx, score in zip(neural_items, neural_scores):
            item_id = self.idx_to_item[item_idx]
            if item_id in combined_scores:
                combined_scores[item_id] += weight_neural * score
            else:
                combined_scores[item_id] = weight_neural * score
        
        # 5. เรียงลำดับและเลือก top-k
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 6. เพิ่มข้อมูลเมนู
        recommendations = []
        for item_id, score in sorted_items[:top_k]:
            menu_info = self.item_features[self.item_features['menu_id'] == item_id].iloc[0]
            recommendations.append({
                'menu_id': item_id,
                'menu_name': menu_info.get('menu_name', 'Unknown'),
                'category': menu_info.get('category', 'Unknown'),
                'price': menu_info.get('price', 0),
                'predicted_score': score,
                'popularity': menu_info.get('popularity', 0)
            })
        
        return recommendations
    
    def _recommend_for_new_user(self, top_k=10):
        """แนะนำเมนูสำหรับลูกค้าใหม่ (based on popularity)"""
        print("👤 ลูกค้าใหม่ - แนะนำเมนูยอดนิยม")
        
        # แนะนำเมนูที่ได้รับความนิยมสูงสุด
        popular_items = self.item_features.nlargest(top_k, 'popularity')
        
        recommendations = []
        for _, item in popular_items.iterrows():
            recommendations.append({
                'menu_id': item['menu_id'],
                'menu_name': item.get('menu_name', 'Unknown'),
                'category': item.get('category', 'Unknown'),
                'price': item['price'],
                'predicted_score': item['popularity'],
                'popularity': item['popularity']
            })
        
        return recommendations
    
    def get_user_profile(self, customer_id):
        """ดึงข้อมูลโปรไฟล์ลูกค้า"""
        user_info = self.user_features[self.user_features['customer_id'] == customer_id]
        if len(user_info) == 0:
            return None
        
        user_info = user_info.iloc[0]
        
        # หาประวัติการสั่งอาหาร
        orders = pd.read_csv('data/orders.csv')
        user_orders = orders[orders['customer_id'] == customer_id]
        
        profile = {
            'customer_id': customer_id,
            'age': user_info['age'],
            'avg_budget': user_info['avg_budget'],
            'total_orders': len(user_orders),
            'favorite_categories': user_orders['category'].value_counts().to_dict(),
            'avg_order_amount': user_orders['amount'].mean() if len(user_orders) > 0 else 0
        }
        
        return profile
    
    def save_models(self, model_dir="models"):
        """บันทึกโมเดล"""
        os.makedirs(model_dir, exist_ok=True)
        print(f"💾 กำลังบันทึกโมเดลใน {model_dir}/...")
        
        # บันทึก Matrix Factorization model
        with open(f'{model_dir}/matrix_model.pkl', 'wb') as f:
            pickle.dump(self.matrix_model, f)
        
        # บันทึก Neural CF model
        if self.neural_model and self.neural_model.model:
            self.neural_model.model.save(f'{model_dir}/neural_model.h5')
        
        print("✅ บันทึกโมเดลเสร็จสิ้น")

# Create alias for backward compatibility 
NeuralCollaborativeFilteringModel = NeuralCollaborativeFiltering

if __name__ == "__main__":
    # สร้างและฝึกโมเดล
    system = HybridRecommendationSystem()
    system.train_models()
    system.save_models()
    
    # ทดสอบการแนะนำ
    print("\n🧪 ทดสอบการแนะนำ:")
    recommendations = system.recommend_for_user('C0001', top_k=5)
    
    print("🍽️ เมนูที่แนะนำ:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['menu_name']} ({rec['category']}) - {rec['price']:.2f} บาท")
        print(f"      คะแนน: {rec['predicted_score']:.3f}")
    
    print("\n🎉 ระบบแนะนำเมนูพร้อมใช้งาน!")
