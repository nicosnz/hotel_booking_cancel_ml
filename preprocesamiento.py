import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix)

RUTA_CSV = os.path.join(os.path.dirname(__file__), 'dataset', 'hotel_bookings.csv')
RESULTADOS = os.path.join(os.path.dirname(__file__), 'resultados')
os.makedirs(RESULTADOS, exist_ok=True)

# Columnas que NO se conocen al momento de hacer la reserva (fuga de información)
FUGA = ['reservation_status',        # 'Canceled' / 'Check-Out' = la respuesta
        'reservation_status_date',   # fecha del último estado (cancelación o salida)
        'assigned_room_type',        # se asigna en el check-in
        'booking_changes',           # cambios acumulados después de reservar
        'days_in_waiting_list']      # tiempo hasta confirmar, posterior a la solicitud
OTRAS_DESCARTADAS = ['arrival_date_year',  # el año no generaliza a reservas futuras
                     'agent', 'company']   # IDs de alta cardinalidad -> se usan indicadores

MESES = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']

CATEGORICAS = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
               'reserved_room_type', 'deposit_type', 'customer_type']
NUMERICAS = ['lead_time', 'arrival_date_month', 'arrival_date_week_number',
             'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
             'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
             'previous_bookings_not_canceled', 'adr', 'required_car_parking_spaces',
             'total_of_special_requests', 'tiene_agente', 'tiene_empresa',
             'total_noches', 'total_huespedes']
TOP_PAISES = 15


def leer_crudo():
    return pd.read_csv(RUTA_CSV)


def limpiar(df):
    df = df.copy()
    # Nulos
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Desconocido')
    df['tiene_agente'] = df['agent'].notna().astype(int)
    df['tiene_empresa'] = df['company'].notna().astype(int)

    # Variables nuevas
    df['total_noches'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_huespedes'] = df['adults'] + df['children'] + df['babies']
    df['arrival_date_month'] = df['arrival_date_month'].map({m: i + 1 for i, m in enumerate(MESES)})

    # Filas inválidas: reservas sin huéspedes y precios imposibles/atípicos
    df = df[(df['total_huespedes'] > 0) & (df['adr'] >= 0) & (df['adr'] <= 5000)]

    # País: los 15 más frecuentes, el resto como 'Otros'
    top = df['country'].value_counts().index[:TOP_PAISES]
    df['country'] = df['country'].where(df['country'].isin(top), 'Otros')

    # Duplicados exactos (bloques de reservas de grupo idénticas): se deja una copia
    # para que la misma reserva no aparezca a la vez en train y en test
    return df.drop(columns=FUGA + OTRAS_DESCARTADAS).drop_duplicates()


def cargar_datos(test_size=0.25, random_state=42):
    df = limpiar(leer_crudo())
    X = df[CATEGORICAS + NUMERICAS]
    y = df['is_canceled']
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def crear_preprocesador():
    # One-hot para categóricas y escalado para numéricas (necesario para SVM,
    # no afecta a los árboles)
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAS),
        ('num', StandardScaler(), NUMERICAS)])


def nombres_variables(pipeline):
    # Nombres de columnas tras el one-hot (para la importancia de variables)
    return [n.split('__', 1)[1] for n in pipeline[0].get_feature_names_out()]


def evaluar(nombre, modelo, X_te, y_te, t_fit, umbral=0.5):
    proba = modelo.predict_proba(X_te)[:, 1]
    yp = (proba > umbral).astype(int)   # ">" igual que predict() en caso de empate
    print(f'\n== {nombre} ==')
    print('Matriz de confusion (filas = real, columnas = predicho):\n', confusion_matrix(y_te, yp))
    return {'Modelo': nombre,
            'Accuracy': accuracy_score(y_te, yp),
            'Precision': precision_score(y_te, yp),
            'Recall': recall_score(y_te, yp),
            'F1': f1_score(y_te, yp),
            'ROC-AUC': roc_auc_score(y_te, proba),
            'Tiempo (s)': t_fit}


def carpeta(nombre):
    # resultados/<nombre>/ : una carpeta por modelo (y otra para el EDA)
    ruta = os.path.join(RESULTADOS, nombre)
    os.makedirs(ruta, exist_ok=True)
    return ruta


def guardar_resultados(clave, metricas, proba):
    # Para que comparacion.py no tenga que reentrenar
    pd.DataFrame([metricas]).to_csv(os.path.join(carpeta(clave), 'metricas.csv'), index=False)
    np.save(os.path.join(carpeta(clave), 'proba_test.npy'), proba)


if __name__ == '__main__':
    crudo = leer_crudo()
    limpio = limpiar(crudo)
    print('Crudo :', crudo.shape)
    print('Limpio:', limpio.shape, f'({len(crudo) - len(limpio)} filas eliminadas)')
    print('Columnas eliminadas:', FUGA + OTRAS_DESCARTADAS)
    X_tr, X_te, y_tr, y_te = cargar_datos()
    print('Train:', X_tr.shape, '| Test:', X_te.shape)
    print('Tasa de cancelacion train/test:', round(y_tr.mean(), 4), round(y_te.mean(), 4))
    print('Nº variables tras one-hot:', crear_preprocesador().fit(X_tr).transform(X_tr[:5]).shape[1])
