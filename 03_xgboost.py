import os, time, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve)
from xgboost import XGBClassifier
from preprocesamiento import (cargar_datos, crear_preprocesador, nombres_variables,
    evaluar, guardar_resultados, carpeta)

X_tr, X_te, y_tr, y_te = cargar_datos()
print('Train:', X_tr.shape, '| Test:', X_te.shape)
SALIDA = carpeta('xgboost')

PARAMS = dict(learning_rate=0.1,      # shrinkage: contribución de cada árbol
              max_depth=6,            # profundidad de cada árbol
              subsample=0.9,          # submuestreo de filas por árbol
              colsample_bytree=0.9,   # submuestreo de columnas por árbol
              reg_lambda=1.0,         # regularización L2
              eval_metric='logloss', random_state=42, n_jobs=-1)

# 1) Early stopping: se separa un 15% de train como validación (el test no se toca)
X_fit, X_val, y_fit, y_val = train_test_split(
    X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)
prep = crear_preprocesador().fit(X_fit)
xgb_es = XGBClassifier(n_estimators=1000, early_stopping_rounds=30, **PARAMS)
t0 = time.time()
xgb_es.fit(prep.transform(X_fit), y_fit,
           eval_set=[(prep.transform(X_fit), y_fit), (prep.transform(X_val), y_val)],
           verbose=False)
t_es = time.time() - t0
n_arboles = xgb_es.best_iteration + 1
print(f'\nEarly stopping: mejor iteracion = {n_arboles} arboles (de 1000 posibles) | {t_es:.1f} s')

# Curva de aprendizaje: logloss en entrenamiento y validación por iteración
hist = xgb_es.evals_result()
plt.figure(figsize=(8, 4.5))
plt.plot(hist['validation_0']['logloss'], label='Entrenamiento')
plt.plot(hist['validation_1']['logloss'], label='Validación')
plt.axvline(xgb_es.best_iteration, color='gray', ls='--', label=f'Mejor iteración ({n_arboles})')
plt.xlabel('Nº de árboles'); plt.ylabel('Log loss'); plt.legend()
plt.title('XGBoost - curva de aprendizaje con early stopping')
plt.tight_layout(); plt.savefig(os.path.join(SALIDA, 'curva_aprendizaje.png'), dpi=150)

# 2) Modelo final: se reentrena con TODO train usando el nº de árboles encontrado
xgb = Pipeline([('prep', crear_preprocesador()),
                ('modelo', XGBClassifier(n_estimators=n_arboles, **PARAMS))])
t0 = time.time(); xgb.fit(X_tr, y_tr); t_xgb = time.time() - t0

m = evaluar(f'XGBoost ({n_arboles} arboles)', xgb, X_te, y_te, t_xgb)
print('Accuracy train:', round(accuracy_score(y_tr, xgb.predict(X_tr)), 4))
guardar_resultados('xgboost', m, xgb.predict_proba(X_te)[:, 1])

# Comparación con los modelos anteriores
previos = [pd.read_csv(os.path.join(carpeta(c), 'metricas.csv')) for c in ['arbol', 'random_forest']]
tabla = pd.concat(previos + [pd.DataFrame([m])]).set_index('Modelo')
cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Tiempo (s)']
print('\nComparacion:\n', tabla[cols].round(4).to_string())

# Validación cruzada (sobre train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1 = cross_val_score(xgb, X_tr, y_tr, cv=cv, scoring='f1')
print('\nXGBoost F1 CV:', f1.round(4), 'media =', round(f1.mean(), 4), '± ', round(f1.std(), 4))

# 3) Umbral de decisión: por defecto 0.5; subirlo reduce falsas alarmas (útil para
#    overbooking), bajarlo detecta más cancelaciones (útil para contactar al cliente)
proba = xgb.predict_proba(X_te)[:, 1]
filas = []
for u in [0.3, 0.4, 0.5, 0.6, 0.7]:
    yp = (proba > u).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
    filas.append({'Umbral': u, 'Accuracy': accuracy_score(y_te, yp),
                  'Precision': precision_score(y_te, yp), 'Recall': recall_score(y_te, yp),
                  'F1': f1_score(y_te, yp), 'Cancelaciones detectadas (TP)': tp,
                  'Falsas alarmas (FP)': fp, 'No detectadas (FN)': fn})
umbrales = pd.DataFrame(filas).set_index('Umbral')
print('\nEfecto del umbral de decision:\n', umbrales.round(4).to_string())

prec, rec, thr = precision_recall_curve(y_te, proba)
plt.figure(figsize=(8, 4.5))
plt.plot(thr, prec[:-1], label='Precision')
plt.plot(thr, rec[:-1], label='Recall')
plt.plot(thr, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12), label='F1', ls='--')
for u in [0.5, 0.7]:
    plt.axvline(u, color='gray', lw=0.8, ls=':')
plt.xlabel('Umbral de decisión'); plt.ylabel('Valor'); plt.ylim(0, 1.02); plt.legend()
plt.title('XGBoost - Precision / Recall según el umbral')
plt.tight_layout(); plt.savefig(os.path.join(SALIDA, 'umbral_precision_recall.png'), dpi=150)

# Importancia de variables (gain: cuánto mejora la pérdida cada variable)
imp = pd.Series(xgb[-1].feature_importances_, index=nombres_variables(xgb)).sort_values(ascending=False)
print('\nTop-15 importancia:\n', imp.head(15).round(4).to_string())
plt.figure(figsize=(8, 6))
imp.head(15).plot(kind='barh', color='#C44E52'); plt.gca().invert_yaxis()
plt.title('Top-15 importancia de variables - XGBoost')
plt.tight_layout(); plt.savefig(os.path.join(SALIDA, 'importancia.png'), dpi=150)
