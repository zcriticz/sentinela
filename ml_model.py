import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class IncidentAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.severity_model = None
        self.response_model = None
        self.cluster_model = None
        self._prepare_data()
    
    def _prepare_data(self):
        self.df['hour'] = pd.to_datetime(self.df['datetime']).dt.hour
        self.df['weekday_num'] = pd.to_datetime(self.df['datetime']).dt.dayofweek
        self.df['month_num'] = pd.to_datetime(self.df['datetime']).dt.month
        
        categorical_cols = ['type', 'subtype', 'neighborhood', 'shift', 'vehicle']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
    
    def train_severity_model(self) -> dict:
        features = ['type_encoded', 'subtype_encoded', 'neighborhood_encoded', 'hour', 'weekday_num', 'month_num']
        
        X = self.df[features]
        y = self.df['severity']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.severity_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.severity_model.fit(X_train, y_train)
        
        y_pred = self.severity_model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        feature_importance = dict(zip(
            ['Tipo', 'Subtipo', 'Bairro', 'Hora', 'Dia da Semana', 'Mês'],
            self.severity_model.feature_importances_
        ))
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def train_response_model(self) -> dict:
        features = ['type_encoded', 'neighborhood_encoded', 'hour', 'weekday_num', 'severity']
        
        X = self.df[features]
        y = self.df['response_time_min']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.response_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.response_model.fit(X_train, y_train)
        
        y_pred = self.response_model.predict(X_test)
        mae = np.mean(np.abs(y_pred - y_test))
        r2 = self.response_model.score(X_test, y_test)
        
        return {'mae': mae, 'r2_score': r2}
    
    def perform_clustering(self, n_clusters: int = 5) -> dict:
        features = ['type_encoded', 'hour', 'weekday_num', 'severity', 'response_time_min', 'victims']
        
        X = self.df[features].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.cluster_model.fit_predict(X_scaled)
        
        self.df['cluster'] = clusters
        
        cluster_analysis = {}
        for i in range(n_clusters):
            mask = self.df['cluster'] == i
            cluster_df = self.df[mask]
            
            cluster_analysis[f'Cluster {i+1}'] = {
                'incident_count': len(cluster_df),
                'main_type': cluster_df['type'].mode().iloc[0] if len(cluster_df) > 0 else 'N/A',
                'avg_severity': cluster_df['severity'].mean(),
                'avg_response_time': cluster_df['response_time_min'].mean(),
                'main_shift': cluster_df['shift'].mode().iloc[0] if len(cluster_df) > 0 else 'N/A',
                'percentage': len(cluster_df) / len(self.df) * 100
            }
        
        silhouette = silhouette_score(X_scaled, clusters)
        
        return {'n_clusters': n_clusters, 'silhouette_score': silhouette, 'cluster_analysis': cluster_analysis}
    
    def predict_demand(self) -> dict:
        self.df['year_month'] = pd.to_datetime(self.df['datetime']).dt.to_period('M')
        monthly_demand = self.df.groupby('year_month').size().reset_index(name='incident_count')
        monthly_demand['year_month'] = monthly_demand['year_month'].astype(str)
        monthly_demand['index'] = range(len(monthly_demand))
        
        if len(monthly_demand) > 2:
            x = monthly_demand['index'].values
            y = monthly_demand['incident_count'].values
            coef = np.polyfit(x, y, 1)
            trend = 'increasing' if coef[0] > 0 else 'decreasing'
            rate = coef[0]
        else:
            trend = 'insufficient'
            rate = 0
        
        return {'monthly_demand': monthly_demand, 'trend': trend, 'monthly_rate': rate}
    
    def analyze_hotspots(self, top_n: int = 5) -> dict:
        neighborhood_stats = self.df.groupby('neighborhood').agg({
            'id': 'count', 'severity': 'mean', 'response_time_min': 'mean', 'victims': 'sum'
        }).rename(columns={'id': 'incident_count'})
        
        neighborhood_stats = neighborhood_stats.sort_values('incident_count', ascending=False)
        
        type_neighborhood = self.df.groupby(['type', 'neighborhood']).size().reset_index(name='incident_count')
        type_neighborhood = type_neighborhood.sort_values('incident_count', ascending=False)
        
        return {
            'top_neighborhoods': neighborhood_stats.head(top_n).to_dict('index'),
            'top_combinations': type_neighborhood.head(top_n * 2).to_dict('records')
        }
    
    def generate_insights(self) -> list:
        insights = []
        
        most_common_type = self.df['type'].mode().iloc[0]
        type_pct = (self.df['type'] == most_common_type).mean() * 100
        insights.append(f"O tipo de ocorrência mais comum é '{most_common_type}', representando {type_pct:.1f}% do total.")
        
        peak_hour = self.df['hour'].mode().iloc[0]
        insights.append(f"O horário com maior número de ocorrências é às {peak_hour}h.")
        
        severe_incidents = self.df[self.df['severity'] >= 4]
        if len(severe_incidents) > 0:
            severe_neighborhood = severe_incidents['neighborhood'].mode().iloc[0]
            insights.append(f"O bairro '{severe_neighborhood}' concentra a maior parte das ocorrências de alta gravidade.")
        
        avg_response = self.df['response_time_min'].mean()
        insights.append(f"O tempo médio de resposta é de {avg_response:.1f} minutos.")
        
        busiest_day = self.df['weekday'].mode().iloc[0]
        insights.append(f"{busiest_day} é o dia da semana com mais ocorrências.")
        
        if self.df['victims'].sum() > 0:
            victims_by_type = self.df.groupby('type')['victims'].sum()
            highest_victims_type = victims_by_type.idxmax()
            insights.append(f"Ocorrências do tipo '{highest_victims_type}' apresentam o maior número de vítimas.")
        
        return insights


def run_full_analysis(df: pd.DataFrame) -> dict:
    analyzer = IncidentAnalyzer(df)
    
    return {
        'severity_model': analyzer.train_severity_model(),
        'response_model': analyzer.train_response_model(),
        'clustering': analyzer.perform_clustering(),
        'demand_forecast': analyzer.predict_demand(),
        'hotspots': analyzer.analyze_hotspots(),
        'insights': analyzer.generate_insights(),
        'analyzed_df': analyzer.df
    }
