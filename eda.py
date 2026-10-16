import os, pandas as pd
import matplotlib.pyplot as plt
from preprocesamiento import leer_crudo, limpiar, carpeta

crudo = leer_crudo()
print('Dimensiones:', crudo.shape)
print('\nNulos por columna:\n', crudo.isna().sum()[crudo.isna().sum() > 0])
print('\nProporcion de clases (is_canceled):\n', crudo['is_canceled'].value_counts(normalize=True).round(3))

df = limpiar(crudo)


def tasa_por(col, ax, top=None):
    tasa = df.groupby(col)['is_canceled'].agg(['mean', 'size']).sort_values('size', ascending=False)
    if top:
        tasa = tasa.head(top)
    tasa = tasa.sort_values('mean')
    ax.barh(tasa.index.astype(str), tasa['mean'], color='#4C72B0')
    ax.axvline(df['is_canceled'].mean(), color='gray', ls='--', lw=1, label='media global')
    ax.set_xlabel('Tasa de cancelación'); ax.set_title(f'Cancelación por {col}')
    ax.legend(loc='lower right')
    print(f'\nTasa de cancelacion por {col}:\n', tasa.round(3).to_string())


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
tasa_por('deposit_type', axes[0, 0])
tasa_por('market_segment', axes[0, 1])
tasa_por('hotel', axes[1, 0])
tasa_por('country', axes[1, 1], top=10)
plt.tight_layout(); plt.savefig(os.path.join(carpeta('eda'), 'tasas_cancelacion.png'), dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for clase, etiqueta in [(0, 'Llegó'), (1, 'Canceló')]:
    axes[0].hist(df.loc[df['is_canceled'] == clase, 'lead_time'], bins=60, alpha=0.6,
                 density=True, label=etiqueta)
axes[0].set_xlabel('lead_time (días de anticipación)'); axes[0].set_title('Anticipación por clase')
axes[0].legend()
pet = df.groupby('total_of_special_requests')['is_canceled'].mean()
axes[1].bar(pet.index.astype(str), pet.values, color='#4C72B0')
axes[1].set_xlabel('Nº de peticiones especiales'); axes[1].set_ylabel('Tasa de cancelación')
axes[1].set_title('Peticiones especiales vs cancelación')
plt.tight_layout(); plt.savefig(os.path.join(carpeta('eda'), 'lead_time_peticiones.png'), dpi=150)
print('\nlead_time medio por clase:\n', df.groupby('is_canceled')['lead_time'].mean().round(1))
