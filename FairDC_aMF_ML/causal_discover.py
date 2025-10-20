import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dowhy import CausalModel
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class OriginalFairDCCausalGraph:
    def __init__(self, train_file, users_features_file, movies_features_file):
        self._load_data(train_file, users_features_file, movies_features_file)
        self.process_data()
        self.causal_matrix = self.build_causal_effect_matrix()
        
    def _load_data(self, train_file, users_features_file, movies_features_file):
        self.movies_features = np.load(movies_features_file)
        self.train_data, _ = np.load(train_file, allow_pickle=True)
        self.users_features = np.load(users_features_file)
        
    def process_data(self):
        records = []
        for user_id, item_list in self.train_data.items():
            user_features = self.users_features[user_id]
            for item_id in item_list:
                record = {
                    'user_id': user_id, 'item_id': item_id,
                    'gender': user_features[0],
                    'age': user_features[1],
                    'occupation': user_features[2]
                }
                for i in range(18):
                    record[f'genre_{i}'] = self.movies_features[item_id][i]
                records.append(record)
        
        self.df = pd.DataFrame(records)
    
    def compute_causal_value(self, user_feature, movie_genre):
        data = self.df.copy()
        treatment = user_feature
        outcome = f'genre_{movie_genre}'
        
        control_features = ['gender', 'age', 'occupation']
        control_features.remove(treatment)
        
        try:
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                common_causes=control_features
            )
            
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.generalized_linear_model",
                method_params={
                    "glm_family": sm.families.Gaussian(),
                    "alpha": 0.05
                }
            )
            return estimate.value
        except Exception as e:
            print(f"Warning: {user_feature}->{outcome}: {str(e)}")
            return 0.0
    
    def build_causal_effect_matrix(self):
        matrix = pd.DataFrame(
            index=['gender', 'age', 'occupation'],
            columns=[f'genre_{i}' for i in range(18)]
        )
        
        total_combinations = 3 * 18
        with tqdm(total=total_combinations, desc="backdoor") as pbar:
            for feature_idx, feature in enumerate(['gender', 'age', 'occupation']):
                print(f"\n compute {feature} causal effects:")
                for genre_idx in range(18):
                    causal_value = self.compute_causal_value(feature, genre_idx)
                    matrix.at[feature, f'genre_{genre_idx}'] = causal_value
                    pbar.set_postfix_str(f"{feature}->genre_{genre_idx}: {causal_value:.4f}")
                    pbar.update(1)
        
        return matrix

class HCFLayer:
    def __init__(self):
        self.mediator_model = None
        self.outcome_model = None
        
    def compute_content_mediator(self, data):
        mediator = []
        for _, row in data.iterrows():
            user_vec = np.array([row['gender'], row['age'], row['occupation']])
            user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)
            
            item_vec = np.array([row[f'genre_{i}'] for i in range(18)])
            item_vec = item_vec / (np.linalg.norm(item_vec) + 1e-8)
            
            content_match = np.dot(user_vec[:3], item_vec[:3])
            mediator.append(content_match)
            
        return np.array(mediator)
    
    def fit_models(self, data, user_feature, genre_col):
        mediator = self.compute_content_mediator(data)
        
        X = []
        for _, row in data.iterrows():
            features = [row['gender'], row['age'], row['occupation']]
            features.extend([row[f'genre_{i}'] for i in range(18)])
            X.append(features)
        
        X = np.array(X)
        self.mediator_model = LinearRegression()
        self.mediator_model.fit(X, mediator)
        
        X_outcome = []
        y_outcome = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            user_val = row[user_feature]
            item_content = [row[f'genre_{j}'] for j in range(18)]
            med_val = mediator[i]
            
            features = [user_val] + item_content + [med_val]
            X_outcome.append(features)
            y_outcome.append(1 if row[genre_col] > 0 else 0)
        
        if len(set(y_outcome)) > 1:
            self.outcome_model = LogisticRegression(random_state=42)
            self.outcome_model.fit(X_outcome, y_outcome)
            return True
        return False
    
    def compute_HCF_effect(self, data):
        if not self.mediator_model or not self.outcome_model:
            return 0
            
        try:
            mediator_samples = np.linspace(-1, 1, 3)
            total_effect = 0
            
            for m_val in mediator_samples:
                sample_features = [1] + [0.5] * 18 + [m_val]
                p_l = self.outcome_model.predict_proba([sample_features])[0][1]
                total_effect += p_l / len(mediator_samples)
                
            return total_effect
        except:
            return 0

class CausalityGraph4ML1M:
    def __init__(self, train_file, users_features_file, movies_features_file):
        print("constructing causal graph...")
        self.fairdc_graph = OriginalFairDCCausalGraph(train_file, users_features_file, movies_features_file)
        
        self.HCF_layer = HCFLayer()
        
        self.causal_matrix = self.build_mixed_causal_matrix()
    
    def build_mixed_causal_matrix(self):
        fairdc_matrix = self.fairdc_graph.causal_matrix.copy()
        
        enhanced_data = self.fairdc_graph.df.copy()
        
        total_enhancements = 3 * 18
        with tqdm(total=total_enhancements, desc="enhence") as pbar:
            for feature in ['gender', 'age', 'occupation']:
                for genre_idx in range(18):
                    genre_col = f'genre_{genre_idx}'
                    
                    genre_data = enhanced_data[enhanced_data[genre_col] == 1]
                    if len(genre_data) >= 30:
                        try:
                            success = self.HCF_layer.fit_models(genre_data, feature, genre_col)
                            if success:
                                HCF_effect = self.HCF_layer.compute_HCF_effect(genre_data)
                                
                                original_effect = fairdc_matrix.at[feature, genre_col]
                                mixed_effect = 0.8 * original_effect + 0.2 * HCF_effect
                                fairdc_matrix.at[feature, genre_col] = mixed_effect
                                
                                pbar.set_postfix_str(f"{feature}->{genre_col}: done")
                            else:
                                pbar.set_postfix_str(f"{feature}->{genre_col}: HCF doing")
                        except:
                            pbar.set_postfix_str(f"{feature}->{genre_col}: error")
                    else:
                        pbar.set_postfix_str(f"{feature}->{genre_col}: insufficient data")
                    
                    pbar.update(1)
        
        return fairdc_matrix
    
    def get_user_item_causal_matrix(self, user_id, item_id):
        try:
            causal_matrix = np.zeros((3, 18))
            movie_features = self.fairdc_graph.movies_features[item_id]
            
            for genre_idx in range(18):
                if movie_features[genre_idx] == 1:
                    causal_matrix[0][genre_idx] = self.causal_matrix.iloc[0][f'genre_{genre_idx}']
                    causal_matrix[1][genre_idx] = self.causal_matrix.iloc[1][f'genre_{genre_idx}']
                    causal_matrix[2][genre_idx] = self.causal_matrix.iloc[2][f'genre_{genre_idx}']
            
            return causal_matrix
        except:
            return np.zeros((3, 18))
    
    def get_batch_user_item_causal_matrices(self, user_item_pairs):
        matrices = []
        for user_id, item_id in user_item_pairs:
            matrix = self.get_user_item_causal_matrix(user_id, item_id)
            matrices.append(matrix)
        return np.array(matrices)

if __name__ == '__main__':
    train_file = "../data/ml/train.npy"
    users_features_file = "../data/ml/users_features.npy"
    movies_features_file = "../data/ml/movie_features.npy"
    
    causal_graph = CausalityGraph4ML1M(train_file, users_features_file, movies_features_file)
    
    print("\ncausal effect matrix:")
    print(causal_graph.causal_matrix)
    
    np.save("../data/ml/fairdc_HCF_mixed_matrix.npy", 
            causal_graph.causal_matrix.values)
    print("saved mixed causal effect matrix")