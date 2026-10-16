import os, time, pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from preprocesamiento import (cargar_datos, crear_preprocesador, nombres_variables,
    evaluar, guardar_resultados, carpeta)

X_tr, X_te, y_tr, y_te = cargar_datos()
print('Train:', X_tr.shape, '| Test:', X_te.shape)


def crear_rf(n):
    return Pipeline([('prep', crear_preprocesador()),
                     ('modelo', RandomForestClassifier(
                         n_estimators=n,         # número de árboles del bosque
                         max_features='sqrt',    # variables aleatorias por división
                         oob_score=True,         # evaluación out-of-bag
                         n_jobs=-1, random_state=42))])


# Random Forest con distinta cantidad de árboles
filas, modelos = [], {}
for n in [100, 200]:
    rf = crear_rf(n)
    t0 = time.time(); rf.fit(X_tr, y_tr); t = time.time() - t0
    modelos[n] = rf
    m = evaluar(f'Random Forest ({n} arboles)', rf, X_te, y_te, t)
    m['OOB'] = rf[-1].oob_score_
    m['Acc. train'] = accuracy_score(y_tr, rf.predict(X_tr))
    filas.append(m)
resultados = pd.DataFrame(filas).set_index('Modelo')
print('\n', resultados.round(4).to_string())

bosque = modelos[200]
# OOB: cada árbol se evalúa con las reservas que no entraron en su muestra bootstrap
print('\nOOB score (200 arboles):', round(bosque[-1].oob_score_, 4))

# Comparación con el árbol base (Paso 1)
arbol = pd.read_csv(os.path.join(carpeta('arbol'), 'metricas.csv')).set_index('Modelo')
print('\nComparacion con el modelo base:\n',
      pd.concat([arbol, resultados.loc[['Random Forest (200 arboles)']]])
        [['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Tiempo (s)']].round(4).to_string())

# Validación cruzada (sobre train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
t0 = time.time()
f1 = cross_val_score(crear_rf(200), X_tr, y_tr, cv=cv, scoring='f1')
print('\nRandom Forest F1 CV:', f1.round(4), 'media =', round(f1.mean(), 4), '± ', round(f1.std(), 4),
      '| tiempo =', round(time.time() - t0, 1), 's')

guardar_resultados('random_forest', resultados.reset_index().iloc[1].to_dict(),
                   bosque.predict_proba(X_te)[:, 1])

# Importancia de variables
imp = pd.Series(bosque[-1].feature_importances_, index=nombres_variables(bosque))
imp = imp.sort_values(ascending=False)
print('\nTop-15 importancia:\n', imp.head(15).round(4).to_string())
plt.figure(figsize=(8, 6))
imp.head(15).plot(kind='barh', color='#55A868'); plt.gca().invert_yaxis()
plt.title('Top-15 importancia de variables - Random Forest')
plt.tight_layout(); plt.savefig(os.path.join(carpeta('random_forest'), 'importancia.png'), dpi=150)
