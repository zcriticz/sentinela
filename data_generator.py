import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

INCIDENT_TYPES = {
    'Incêndio': {'weight': 0.20, 'subtypes': ['Residencial', 'Veicular', 'Vegetação', 'Comercial', 'Industrial'], 'avg_service_time': 45, 'avg_severity': 3.5},
    'Pré-Hospitalar': {'weight': 0.35, 'subtypes': ['Acidente de Trânsito', 'Mal Súbito', 'Queda', 'Agressão', 'Afogamento'], 'avg_service_time': 30, 'avg_severity': 3.0},
    'Salvamento': {'weight': 0.25, 'subtypes': ['Pessoa', 'Animal', 'Veículo', 'Árvore', 'Elevador'], 'avg_service_time': 40, 'avg_severity': 2.5},
    'Produtos Perigosos': {'weight': 0.05, 'subtypes': ['Vazamento de Gás', 'Químico', 'Radioativo', 'Biológico'], 'avg_service_time': 60, 'avg_severity': 4.0},
    'Prevenção': {'weight': 0.10, 'subtypes': ['Vistoria', 'Treinamento', 'Orientação', 'Fiscalização'], 'avg_service_time': 120, 'avg_severity': 1.0},
    'Atividade Comunitária': {'weight': 0.05, 'subtypes': ['Palestra', 'Simulado', 'Evento', 'Visita Escolar'], 'avg_service_time': 90, 'avg_severity': 1.0}
}

NEIGHBORHOODS = [
    'Boa Viagem', 'Casa Forte', 'Espinheiro', 'Boa Vista', 'Derby',
    'Graças', 'Madalena', 'Torre', 'Aflitos', 'Pina',
    'Recife Antigo', 'Santo Amaro', 'Casa Amarela', 'Várzea', 'Ibura',
    'Olinda Centro', 'Jaboatão Centro', 'Paulista', 'Cabo', 'Camaragibe'
]

VEHICLES = ['ABT-01', 'ABT-02', 'ABT-03', 'ASE-01', 'ASE-02', 'UR-01', 'UR-02', 'UR-03', 'AEM-01', 'ABTR-01']

SHIFTS = ['Manhã (06h-14h)', 'Tarde (14h-22h)', 'Noite (22h-06h)']

COORDINATE_OFFSETS = {
    'Boa Viagem': (0.035, 0.015), 'Casa Forte': (-0.015, -0.005), 'Espinheiro': (-0.008, -0.008),
    'Boa Vista': (0.002, 0.002), 'Derby': (-0.005, -0.003), 'Graças': (-0.012, -0.006),
    'Madalena': (0.008, -0.015), 'Torre': (0.005, -0.010), 'Aflitos': (-0.010, -0.005),
    'Pina': (0.025, 0.020), 'Recife Antigo': (0.010, 0.015), 'Santo Amaro': (0.003, 0.005),
    'Casa Amarela': (-0.025, -0.010), 'Várzea': (-0.020, -0.035), 'Ibura': (0.045, -0.025),
    'Olinda Centro': (-0.035, 0.010), 'Jaboatão Centro': (0.070, -0.040),
    'Paulista': (-0.065, 0.025), 'Cabo': (0.150, -0.050), 'Camaragibe': (-0.030, -0.050)
}

BASE_LAT, BASE_LON = -8.0476, -34.8770


def generate_coordinates(neighborhood: str) -> tuple:
    offset = COORDINATE_OFFSETS.get(neighborhood, (0, 0))
    lat = BASE_LAT + offset[0] + np.random.uniform(-0.008, 0.008)
    lon = BASE_LON + offset[1] + np.random.uniform(-0.008, 0.008)
    return round(lat, 6), round(lon, 6)


def generate_incidents(n_records: int = 1000, year: int = 2024) -> pd.DataFrame:
    records = []
    
    for i in range(n_records):
        types = list(INCIDENT_TYPES.keys())
        weights = [INCIDENT_TYPES[t]['weight'] for t in types]
        incident_type = np.random.choice(types, p=weights)
        
        config = INCIDENT_TYPES[incident_type]
        subtype = random.choice(config['subtypes'])

        day_of_year = int(np.random.randint(1, 366))
        hour = int(np.random.randint(0, 24))
        minute = int(np.random.randint(0, 60))

        if incident_type == 'Incêndio' and subtype == 'Vegetação':
            day_of_year = int(np.random.choice(
                range(1, 366),
                p=np.array([0.5 if 150 <= d <= 270 else 1.0 for d in range(1, 366)]) / 
                  sum([0.5 if 150 <= d <= 270 else 1.0 for d in range(1, 366)])
            ))
        
        incident_datetime = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour, minutes=minute)
        
        if 6 <= hour < 14:
            shift = 'Manhã (06h-14h)'
        elif 14 <= hour < 22:
            shift = 'Tarde (14h-22h)'
        else:
            shift = 'Noite (22h-06h)'

        neighborhood = random.choice(NEIGHBORHOODS)
        lat, lon = generate_coordinates(neighborhood)

        response_time = max(5, int(np.random.normal(15, 5)))
        service_time = max(10, int(np.random.normal(config['avg_service_time'], 15)))
        severity = min(5, max(1, int(np.random.normal(config['avg_severity'], 1))))
        
        victims = 0
        if incident_type in ['Incêndio', 'Pré-Hospitalar', 'Salvamento'] and severity >= 3:
            victims = int(np.random.poisson(severity - 2))

        vehicle = random.choice(VEHICLES)
        disaster_related = random.random() < 0.05
        
        outcomes = ['Resolvido no Local', 'Encaminhado Hospital', 'Falso Alarme', 'Cancelado']
        outcome_weights = [0.6, 0.25, 0.1, 0.05]
        outcome = np.random.choice(outcomes, p=outcome_weights)
        
        record = {
            'id': f'OC-{year}-{i+1:05d}',
            'datetime': incident_datetime,
            'date': incident_datetime.date(),
            'time': incident_datetime.time(),
            'month': incident_datetime.month,
            'weekday': incident_datetime.strftime('%A'),
            'shift': shift,
            'type': incident_type,
            'subtype': subtype,
            'neighborhood': neighborhood,
            'latitude': lat,
            'longitude': lon,
            'vehicle': vehicle,
            'response_time_min': response_time,
            'service_time_min': service_time,
            'severity': severity,
            'victims': victims,
            'disaster_related': disaster_related,
            'outcome': outcome
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    weekday_map = {
        'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
        'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['weekday'] = df['weekday'].map(weekday_map)
    
    return df.sort_values('datetime').reset_index(drop=True)


def generate_historical_data(years: list = None) -> pd.DataFrame:
    if years is None:
        years = [2022, 2023, 2024]
    
    dfs = []
    for year in years:
        n = int(1000 * (1 + (year - 2022) * 0.1))
        df_year = generate_incidents(n_records=n, year=year)
        dfs.append(df_year)
    
    return pd.concat(dfs, ignore_index=True).sort_values('datetime').reset_index(drop=True)


if __name__ == '__main__':
    df = generate_historical_data()
    df.to_csv('incidents.csv', index=False)
    print(f"Generated {len(df)} incident records")
