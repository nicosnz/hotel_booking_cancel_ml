import os, time, pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from preprocesamiento import (cargar_datos, crear_preprocesador, evaluar,
    guardar_resultados, carpeta)

X_tr, X_te, y_tr, y_te = cargar_datos()
SALIDA = carpeta('svm')
N_MUESTRA = 10000

# El SVM con kernel calcula similitudes entre pares de ejemplos: el entrenamiento
# cuesta entre O(n²) y O(n³). Con 63.000 reservas sería muy lento, así que se usa
# una muestra estratificada de train. El test sigue siendo el mismo (completo).
X_m, _, y_m, _ = train_test_split(X_tr, y_tr, train_size=N_MUESTRA, stratify=y_tr, random_state=42)
print('Muestra de entrenamiento:', X_m.shape, '| tasa de cancelacion:', round(y_m.mean(), 4))
print('Test (completo):', X_te.shape)

# 1) Búsqueda de hiperparámetros sobre la muestra
#    C: penalización de errores (alto = margen estrecho, riesgo de sobreajuste)
#    gamma: alcance del kernel RBF (alto = fronteras muy curvas)
svm = Pipeline([('prep', crear_preprocesador()),
                ('modelo', SVC(kernel='rbf'))])
grid = GridSearchCV(svm,
                    {'modelo__C': [0.1, 1, 10, 100],
                     'modelo__gamma': ['scale', 0.01, 0.1]},
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                    scoring='f1', n_jobs=-1)
t0 = time.time(); grid.fit(X_m, y_m); t_grid = time.time() - t0
print(f'\nGridSearch: {t_grid:.1f} s | mejores parametros: {grid.best_params_} | F1 CV = {grid.best_score_:.4f}')

tabla_grid = pd.DataFrame(grid.cv_results_)[['param_modelo__C', 'param_modelo__gamma',
                                             'mean_test_score', 'std_test_score', 'mean_fit_time']]
tabla_grid.columns = ['C', 'gamma', 'F1 CV', 'Desv.', 'Tiempo fit (s)']
print(tabla_grid.sort_values('F1 CV', ascending=False).round(4).to_string(index=False))

# Mapa de calor del grid
piv = tabla_grid.assign(gamma=tabla_grid['gamma'].astype(str)).pivot(index='C', columns='gamma', values='F1 CV')
plt.figure(figsize=(6, 4.5))
plt.imshow(piv.values, cmap='Blues')
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        plt.text(j, i, f'{piv.values[i, j]:.3f}', ha='center', va='center',
                 color='white' if piv.values[i, j] > 0.35 else 'black')
plt.xticks(range(piv.shape[1]), piv.columns); plt.yticks(range(piv.shape[0]), piv.index)
plt.xlabel('gamma'); plt.ylabel('C'); plt.colorbar(label='F1 (CV)')
plt.title('SVM RBF - F1 en validación cruzada')
plt.tight_layout(); plt.savefig(os.path.join(SALIDA, 'grid_hiperparametros.png'), dpi=150)

# 2) Modelo final con los mejores parámetros. El SVM no da probabilidades: se calibran
#    con una sigmoide (Platt) ajustada por CV interna, para el ROC-AUC y la comparación final
mejores = {k.replace('modelo__', ''): v for k, v in grid.best_params_.items()}
svm_final = Pipeline([('prep', crear_preprocesador()),
                      ('modelo', CalibratedClassifierCV(SVC(kernel='rbf', **mejores),
                                                        method='sigmoid', ensemble=False))])
t0 = time.time(); svm_final.fit(X_m, y_m); t_svm = time.time() - t0
m = evaluar(f'SVM RBF (muestra {N_MUESTRA})', svm_final, X_te, y_te, t_svm)
print('Vectores de soporte:', svm_final[-1].calibrated_classifiers_[0].estimator.n_support_.sum(), 'de', N_MUESTRA)
print('Accuracy train (muestra):', round(accuracy_score(y_m, svm_final.predict(X_m)), 4))
guardar_resultados('svm', m, svm_final.predict_proba(X_te)[:, 1])

previos = [pd.read_csv(os.path.join(carpeta(c), 'metricas.csv')) for c in ['arbol', 'random_forest', 'xgboost']]
tabla = pd.concat(previos + [pd.DataFrame([m])]).set_index('Modelo')
print('\nComparacion:\n', tabla[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Tiempo (s)']]
      .round(4).to_string())

# 3) ¿Es peor por el algoritmo o por tener menos datos? XGBoost con la MISMA muestra
xgb_m = Pipeline([('prep', crear_preprocesador()),
                  ('modelo', XGBClassifier(n_estimators=382, learning_rate=0.1, max_depth=6,
                                           subsample=0.9, colsample_bytree=0.9,
                                           random_state=42, n_jobs=-1))])
t0 = time.time(); xgb_m.fit(X_m, y_m); t_xgb_m = time.time() - t0
m_xgb = evaluar(f'XGBoost (muestra {N_MUESTRA})', xgb_m, X_te, y_te, t_xgb_m)
print('\nMisma muestra de 10.000:\n', pd.DataFrame([m, m_xgb]).set_index('Modelo').round(4).to_string())

# 4) Escalabilidad: tiempo de entrenamiento del SVM según el tamaño de la muestra
filas = []
for n in [2500, 5000, 10000, 20000]:
    X_n, _, y_n, _ = train_test_split(X_tr, y_tr, train_size=n, stratify=y_tr, random_state=42)
    s = Pipeline([('prep', crear_preprocesador()), ('modelo', SVC(kernel='rbf', **mejores))])
    t0 = time.time(); s.fit(X_n, y_n); t = time.time() - t0
    filas.append({'n': n, 'Tiempo fit (s)': t, 'F1 test': f1_score(y_te, s.predict(X_te))})
    print(f'n = {n:6d} | fit = {t:6.1f} s | F1 test = {filas[-1]["F1 test"]:.4f}')
esc = pd.DataFrame(filas)

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(esc['n'], esc['Tiempo fit (s)'], 'o-', color='#4C72B0', label='Tiempo de entrenamiento')
ax1.set_xlabel('Tamaño de la muestra de entrenamiento'); ax1.set_ylabel('Tiempo (s)', color='#4C72B0')
ax2 = ax1.twinx()
ax2.plot(esc['n'], esc['F1 test'], 's--', color='#C44E52', label='F1 en test')
ax2.set_ylabel('F1 en test', color='#C44E52')
plt.title('SVM RBF - escalabilidad según el tamaño de la muestra')
fig.tight_layout(); plt.savefig(os.path.join(SALIDA, 'escalabilidad.png'), dpi=150)
