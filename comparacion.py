import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix)
from preprocesamiento import cargar_datos, carpeta

_, _, _, y_te = cargar_datos()   # mismo test que usaron los 4 modelos
SALIDA = carpeta('comparacion')

# Color fijo por modelo (paleta categórica validada, en orden)
MODELOS = {'arbol':         ('Árbol de decisión', '#2a78d6'),
           'random_forest': ('Random Forest',     '#eb6834'),
           'xgboost':       ('XGBoost',           '#1baf7a'),
           'svm':           ('SVM (muestra)',     '#eda100')}
TINTA, TINTA_2, REJILLA = '#1f1f1e', '#5f5e58', '#e4e3df'
plt.rcParams.update({'axes.edgecolor': REJILLA, 'axes.labelcolor': TINTA_2,
                     'xtick.color': TINTA_2, 'ytick.color': TINTA_2, 'text.color': TINTA,
                     'axes.spines.top': False, 'axes.spines.right': False})

metricas = pd.concat([pd.read_csv(os.path.join(carpeta(c), 'metricas.csv')).assign(clave=c)
                      for c in MODELOS]).set_index('clave')
proba = {c: np.load(os.path.join(carpeta(c), 'proba_test.npy')) for c in MODELOS}
metricas['PR-AUC'] = [average_precision_score(y_te, proba[c]) for c in metricas.index]

# 1) Tabla final, ordenada por F1
cols = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'Tiempo (s)']
tabla = metricas.sort_values('F1', ascending=False)[cols]
print('Comparacion final (test = %d reservas, %.1f%% canceladas):' % (len(y_te), 100 * y_te.mean()))
print(tabla.round(4).to_string(index=False))
tabla.round(4).to_csv(os.path.join(SALIDA, 'tabla_comparativa.csv'), index=False)
print('\nReferencia: predecir siempre "no cancela" -> accuracy = %.4f, recall = 0' % (1 - y_te.mean()))

# 2) Curvas ROC y Precision-Recall
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for c, (nombre, color) in MODELOS.items():
    fpr, tpr, _ = roc_curve(y_te, proba[c])
    ax1.plot(fpr, tpr, color=color, lw=2, label=f'{nombre} (AUC = {metricas.loc[c, "ROC-AUC"]:.3f})')
    prec, rec, _ = precision_recall_curve(y_te, proba[c])
    ax2.plot(rec, prec, color=color, lw=2, label=f'{nombre} (AP = {metricas.loc[c, "PR-AUC"]:.3f})')
ax1.plot([0, 1], [0, 1], color=TINTA_2, lw=1, ls='--', label='Azar (AUC = 0.5)')
ax1.set_xlabel('Tasa de falsas alarmas (FPR)'); ax1.set_ylabel('Cancelaciones detectadas (TPR = recall)')
ax1.set_title('Curva ROC')
ax2.axhline(y_te.mean(), color=TINTA_2, lw=1, ls='--', label=f'Azar ({y_te.mean():.3f})')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('Curva Precision-Recall')
for ax in (ax1, ax2):
    ax.grid(color=REJILLA, lw=0.8); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc='lower right' if ax is ax1 else 'upper right', frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(SALIDA, 'curvas_roc_pr.png'), dpi=150)

# 3) Métricas agrupadas por modelo
nombres_m = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
x = np.arange(len(nombres_m)); ancho = 0.2
fig, ax = plt.subplots(figsize=(11, 5))
for i, (c, (nombre, color)) in enumerate(MODELOS.items()):
    vals = metricas.loc[c, nombres_m].values.astype(float)
    barras = ax.bar(x + (i - 1.5) * ancho, vals, ancho * 0.92, color=color, label=nombre)
    ax.bar_label(barras, fmt='%.2f', fontsize=7.5, padding=2, color=TINTA_2)
ax.set_xticks(x, nombres_m); ax.set_ylim(0, 1.05); ax.set_ylabel('Valor en test')
ax.grid(axis='y', color=REJILLA, lw=0.8); ax.set_axisbelow(True)
ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.08))
ax.set_title('Comparación de métricas en test (umbral 0.5)')
plt.tight_layout(); plt.savefig(os.path.join(SALIDA, 'metricas.png'), dpi=150)

# 4) Matrices de confusión (umbral 0.5)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, (c, (nombre, _)) in zip(axes, MODELOS.items()):
    cm = confusion_matrix(y_te, (proba[c] > 0.5).astype(int))
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.sum())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}'.replace(',', '.'), ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > cm.sum() * 0.4 else TINTA)
    ax.set_xticks([0, 1], ['Llega', 'Cancela']); ax.set_yticks([0, 1], ['Llegó', 'Canceló'])
    ax.set_xlabel('Predicho'); ax.set_title(nombre)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel('Real')
plt.tight_layout(); plt.savefig(os.path.join(SALIDA, 'matrices_confusion.png'), dpi=150)

# 5) Uso para el hotel con el mejor modelo según ROC-AUC
mejor = metricas['ROC-AUC'].idxmax()
p = proba[mejor]
print(f'\nMejor modelo (ROC-AUC): {MODELOS[mejor][0]}')
print('\nSegmentacion de las reservas de test por riesgo:')
riesgo = pd.cut(p, [0, 0.3, 0.7, 1], labels=['Bajo (<0.3)', 'Medio (0.3-0.7)', 'Alto (>0.7)'],
                include_lowest=True)
seg = pd.DataFrame({'riesgo': riesgo, 'cancela': y_te.values}).groupby('riesgo', observed=False)['cancela']
seg = pd.DataFrame({'Reservas': seg.size(), '% del total': 100 * seg.size() / len(p),
                    'Tasa real de cancelacion': seg.mean()})
print(seg.round(3).to_string())
seg.round(4).to_csv(os.path.join(SALIDA, 'segmentos_riesgo.csv'))
