import os, time, pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from preprocesamiento import (cargar_datos, crear_preprocesador, nombres_variables,
    evaluar, guardar_resultados, carpeta)

X_tr, X_te, y_tr, y_te = cargar_datos()
print('Train:', X_tr.shape, '| Test:', X_te.shape)

# Árbol sin límite de profundidad: crece hasta memorizar el entrenamiento
arbol_libre = Pipeline([('prep', crear_preprocesador()),
                        ('modelo', DecisionTreeClassifier(random_state=42))])
t0 = time.time(); arbol_libre.fit(X_tr, y_tr); t_libre = time.time() - t0

# Árbol regularizado: limita profundidad y tamaño mínimo de las hojas
arbol = Pipeline([('prep', crear_preprocesador()),
                  ('modelo', DecisionTreeClassifier(
                      max_depth=10,          # profundidad máxima
                      min_samples_leaf=20,   # mínimo de reservas por hoja
                      random_state=42))])
t0 = time.time(); arbol.fit(X_tr, y_tr); t_arbol = time.time() - t0

# Sobreajuste: accuracy en train vs test
print('\nSobreajuste (accuracy train vs test):')
for nombre, m in [('Arbol sin limite', arbol_libre), ('Arbol max_depth=10', arbol)]:
    print(f'  {nombre:20s} train = {accuracy_score(y_tr, m.predict(X_tr)):.4f} | '
          f'test = {accuracy_score(y_te, m.predict(X_te)):.4f} | '
          f'profundidad = {m[-1].get_depth()} | hojas = {m[-1].get_n_leaves()}')

# Evaluación en test
resultados = pd.DataFrame([evaluar('Arbol sin limite', arbol_libre, X_te, y_te, t_libre),
                           evaluar('Arbol max_depth=10', arbol, X_te, y_te, t_arbol)]).set_index('Modelo')
print('\n', resultados.round(4).to_string())

# Validación cruzada (sobre train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for nombre, m in [('Arbol sin limite', arbol_libre), ('Arbol max_depth=10', arbol)]:
    f1 = cross_val_score(m, X_tr, y_tr, cv=cv, scoring='f1', n_jobs=-1)
    print(nombre, 'F1 CV:', f1.round(4), 'media =', round(f1.mean(), 4), '± ', round(f1.std(), 4))

# Se guarda el árbol regularizado como modelo base para la comparación final
guardar_resultados('arbol', resultados.reset_index().iloc[1].to_dict(),
                   arbol.predict_proba(X_te)[:, 1])

# Importancia de variables
nombres = nombres_variables(arbol)
imp = pd.Series(arbol[-1].feature_importances_, index=nombres).sort_values(ascending=False)
print('\nTop-15 importancia:\n', imp.head(15).round(4).to_string())
plt.figure(figsize=(8, 6))
imp.head(15).plot(kind='barh'); plt.gca().invert_yaxis()
plt.title('Top-15 importancia de variables - Árbol de decisión')
plt.tight_layout(); plt.savefig(os.path.join(carpeta('arbol'), 'importancia.png'), dpi=150)

# Primeros 3 niveles del árbol (para interpretar las reglas)
plt.figure(figsize=(22, 9))
plot_tree(arbol[-1], max_depth=3, feature_names=nombres, class_names=['Llega', 'Cancela'],
          filled=True, rounded=True, impurity=False, proportion=True, fontsize=8)
plt.title('Árbol de decisión (primeros 3 niveles). Numéricas estandarizadas')
plt.tight_layout(); plt.savefig(os.path.join(carpeta('arbol'), 'arbol_niveles.png'), dpi=150)
