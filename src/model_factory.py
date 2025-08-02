"""
Model Factory for AI Menu Recommendation System
สร้างและจัดการโมเดลต่างๆ สำหรับระบบแนะนำเมนู
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

class MatrixFactorizationModel:
    """Matrix Factorization Model using NMF"""
    
    def __init__(self, n_components=50, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.user_features = None
        self.item_features = None
        self.user_mapping = None
        self.item_mapping = None
        
    def fit(self, user_item_matrix):
        """Train the model"""
        print("🔥 Training Matrix Factorization Model...")
        
        # Handle missing values
        matrix = user_item_matrix.fillna(0)
        
        # Store mappings
        self.user_mapping = {idx: user_id for idx, user_id in enumerate(matrix.index)}
        self.item_mapping = {idx: item_id for idx, item_id in enumerate(matrix.columns)}
        
        # Train NMF
        self.model = NMF(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=500,
            alpha_W=0.01,
            alpha_H=0.01
        )
        
        self.user_features = self.model.fit_transform(matrix.values)
        self.item_features = self.model.components_
        
        print("✅ Matrix Factorization training completed!")
        
    def predict_for_user(self, user_idx, top_k=10):
        """Predict recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get user features
        user_vector = self.user_features[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(user_vector, self.item_features)
        
        # Get top-k items
        top_items_idx = np.argsort(scores)[::-1][:top_k]
        
        # Map back to item IDs
        recommendations = []
        for idx in top_items_idx:
            item_id = self.item_mapping.get(idx, f"item_{idx}")
            score = scores[idx]
            recommendations.append((item_id, score))
        
        return recommendations
    
    def get_user_item_score(self, user_idx, item_idx):
        """Get prediction score for a specific user-item pair"""
        user_vector = self.user_features[user_idx]
        item_vector = self.item_features[:, item_idx]
        return np.dot(user_vector, item_vector)


class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering Model"""
    
    def __init__(self, num_users, num_items, embedding_size=50, hidden_units=[128, 64]):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.model = None
        self.user_mapping = None
        self.item_mapping = None
        
    def build_model(self):
        """Build NCF model architecture"""
        # Input layers
        user_input = Input(shape=(), name='user_input')
        item_input = Input(shape=(), name='item_input')
        
        # Embedding layers
        user_embedding = Embedding(
            self.num_users, 
            self.embedding_size,
            embeddings_regularizer=l2(1e-6),
            name='user_embedding'
        )(user_input)
        
        item_embedding = Embedding(
            self.num_items, 
            self.embedding_size,
            embeddings_regularizer=l2(1e-6),
            name='item_embedding'
        )(item_input)
        
        # Flatten embeddings
        user_vec = Flatten(name='user_flatten')(user_embedding)
        item_vec = Flatten(name='item_flatten')(item_embedding)
        
        # Concatenate user and item embeddings
        concat = Concatenate(name='concat')([user_vec, item_vec])
        
        # Hidden layers
        x = concat
        for units in self.hidden_units:
            x = Dense(units, activation='relu', kernel_regularizer=l2(1e-6))(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='rating')(x)
        
        # Create model
        self.model = Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def prepare_data(self, user_item_matrix):
        """Prepare data for training"""
        # Create mappings
        users = list(user_item_matrix.index)
        items = list(user_item_matrix.columns)
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(items)}
        
        # Create training data
        user_ids = []
        item_ids = []
        ratings = []
        
        for user_id in users:
            for item_id in items:
                rating = user_item_matrix.loc[user_id, item_id]
                if not pd.isna(rating) and rating > 0:
                    user_ids.append(self.user_mapping[user_id])
                    item_ids.append(self.item_mapping[item_id])
                    ratings.append(min(rating / 5.0, 1.0))  # Normalize to [0,1]
        
        return np.array(user_ids), np.array(item_ids), np.array(ratings)
    
    def fit(self, user_item_matrix, epochs=50, batch_size=256, validation_split=0.2):
        """Train the NCF model"""
        print("🔥 Training Neural Collaborative Filtering Model...")
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Prepare training data
        user_ids, item_ids, ratings = self.prepare_data(user_item_matrix)
        
        # Train model
        history = self.model.fit(
            [user_ids, item_ids], 
            ratings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        print("✅ Neural CF training completed!")
        return history
    
    def predict_for_user(self, user_idx, top_k=10):
        """Predict recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get all items for this user
        user_array = np.full(self.num_items, user_idx)
        item_array = np.arange(self.num_items)
        
        # Predict scores
        scores = self.model.predict([user_array, item_array]).flatten()
        
        # Get top-k items
        top_items_idx = np.argsort(scores)[::-1][:top_k]
        
        # Map back to item IDs
        reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        recommendations = []
        for idx in top_items_idx:
            item_id = reverse_item_mapping.get(idx, f"item_{idx}")
            score = scores[idx]
            recommendations.append((item_id, score))
        
        return recommendations


class HybridRecommendationModel:
    """Hybrid model combining multiple approaches"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.is_fitted = False
        
    def fit(self, user_item_matrix):
        """Train all component models"""
        print("🔥 Training Hybrid Recommendation Model...")
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}...")
            model.fit(user_item_matrix)
        
        self.is_fitted = True
        print("✅ Hybrid model training completed!")
    
    def predict_for_user(self, user_idx, top_k=10):
        """Get hybrid recommendations"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet!")
        
        # Get recommendations from each model
        all_recommendations = {}
        
        for model, weight in zip(self.models, self.weights):
            recommendations = model.predict_for_user(user_idx, top_k * 2)  # Get more for mixing
            for item_id, score in recommendations:
                if item_id not in all_recommendations:
                    all_recommendations[item_id] = 0
                all_recommendations[item_id] += score * weight
        
        # Sort by combined score
        sorted_items = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:top_k]


class ModelFactory:
    """Factory class for creating recommendation models"""
    
    @staticmethod
    def create_matrix_factorization(n_components=50, **kwargs):
        """Create Matrix Factorization model"""
        return MatrixFactorizationModel(n_components=n_components, **kwargs)
    
    @staticmethod
    def create_neural_cf(num_users, num_items, embedding_size=50, **kwargs):
        """Create Neural Collaborative Filtering model"""
        return NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            embedding_size=embedding_size,
            **kwargs
        )
    
    @staticmethod
    def create_hybrid_model(user_item_matrix, model_configs=None):
        """Create hybrid model with multiple components"""
        if model_configs is None:
            model_configs = [
                {'type': 'matrix_factorization', 'params': {'n_components': 50}},
                {'type': 'neural_cf', 'params': {'embedding_size': 32}}
            ]
        
        models = []
        
        for config in model_configs:
            if config['type'] == 'matrix_factorization':
                model = ModelFactory.create_matrix_factorization(**config['params'])
            elif config['type'] == 'neural_cf':
                params = config['params'].copy()
                params['num_users'] = len(user_item_matrix.index)
                params['num_items'] = len(user_item_matrix.columns)
                model = ModelFactory.create_neural_cf(**params)
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            
            models.append(model)
        
        weights = [config.get('weight', 1.0) for config in model_configs]
        return HybridRecommendationModel(models, weights)
    
    @staticmethod
    def optimize_hyperparameters(user_item_matrix, model_type='matrix_factorization'):
        """Optimize hyperparameters using grid search"""
        print(f"🔍 Optimizing hyperparameters for {model_type}...")
        
        if model_type == 'matrix_factorization':
            # Grid search for Matrix Factorization
            param_grid = {
                'n_components': [20, 50, 100],
                'alpha_W': [0.01, 0.1],
                'alpha_H': [0.01, 0.1]
            }
            
            best_score = -np.inf
            best_params = None
            
            for n_comp in param_grid['n_components']:
                for alpha_w in param_grid['alpha_W']:
                    for alpha_h in param_grid['alpha_H']:
                        try:
                            model = NMF(
                                n_components=n_comp,
                                alpha_W=alpha_w,
                                alpha_H=alpha_h,
                                random_state=42,
                                max_iter=200
                            )
                            
                            matrix = user_item_matrix.fillna(0)
                            W = model.fit_transform(matrix.values)
                            H = model.components_
                            
                            # Simple reconstruction error as score
                            reconstructed = np.dot(W, H)
                            score = -np.mean((matrix.values - reconstructed) ** 2)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'n_components': n_comp,
                                    'alpha_W': alpha_w,
                                    'alpha_H': alpha_h
                                }
                        except:
                            continue
            
            print(f"✅ Best parameters: {best_params}")
            return best_params
        
        else:
            print("⚠️ Hyperparameter optimization not implemented for this model type")
            return {}


# Convenience functions
def create_simple_matrix_factorization(user_item_matrix, n_components=50):
    """Create and train a simple Matrix Factorization model"""
    model = ModelFactory.create_matrix_factorization(n_components=n_components)
    model.fit(user_item_matrix)
    return model

def create_simple_neural_cf(user_item_matrix, embedding_size=50):
    """Create and train a simple Neural CF model"""
    num_users = len(user_item_matrix.index)
    num_items = len(user_item_matrix.columns)
    
    model = ModelFactory.create_neural_cf(
        num_users=num_users,
        num_items=num_items,
        embedding_size=embedding_size
    )
    model.fit(user_item_matrix, epochs=30)
    return model

def create_simple_hybrid(user_item_matrix):
    """Create and train a simple hybrid model"""
    model = ModelFactory.create_hybrid_model(user_item_matrix)
    model.fit(user_item_matrix)
    return model
