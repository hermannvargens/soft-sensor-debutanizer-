import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features import IndustrialFeaturePipeline

# ==============================================================================
# CONFIGURAÇÃO DE DIAGNÓSTICO: ESCOLHA A LINHA DO CSV QUE DESEJA AVALIAR
# ==============================================================================
INDICE_MODELO = 44  # Mude para o índice da linha do CSV que você quer estudar

# 1. Carregar os resultados da otimização
caminho_csv = "data/processed/resultado_otimizacao_pls.csv"
if not os.path.exists(caminho_csv):
    print(f"Erro: O arquivo {caminho_csv} não foi encontrado. Rode a otimização primeiro.")
    sys.exit(1)

df_res = pd.read_csv(caminho_csv)

if INDICE_MODELO >= len(df_res):
    print(f"Erro: O índice {INDICE_MODELO} está fora dos limites do CSV (máximo: {len(df_res)-1}).")
    sys.exit(1)

# Isolar o pipeline escolhido
best = df_res.iloc[INDICE_MODELO]

# Extração dinâmica dos lags armazenados no CSV para esta linha específica
colunas_lags = [col for col in df_res.columns if col.startswith('Lag_')]
lags_dinamicos = {col.replace('Lag_', ''): int(best[col]) for col in colunas_lags}

print("="*75)
print(f"AVALIANDO MODELO PLS - LINHA [{INDICE_MODELO}] DO CSV DE OTIMIZAÇÃO")
print("="*75)
print(f"• Configuração: EMA_Span={int(best['EMA_Span'])} | Derivadas={best['Derivadas']} | "
      f"Rolling={best['Rolling']} | Janela={int(best['Roll_Window']) if best['Rolling'] else 'N/A'} | "
      f"Quadráticos={best['Squared']}")
print(f"• Complexidade:  {int(best['N_Features'])} Atributos | {int(best['Best_C'])} Componentes Latentes")
print(f"• Razão C/F:     {best['Best_C']/best['N_Features']:.4f}")
print(f"• R² do Grid:    {best['R2']:.4f} | RMSE: {best['RMSE']:.6f}")
print("\n• Lags Dinâmicos Utilizados nesta Linha:")
for sensor, lag_val in lags_dinamicos.items():
    print(f"  - {sensor}: {lag_val} passos")
print("="*75)

# 2. Carregar dados reais da planta
caminho_dados = "/home/hermann/projeto_soft_sensor/data/raw/debutanizer_data.txt"
data = np.loadtxt(caminho_dados, skiprows=5)
df_raw = pd.DataFrame(data, columns=['Top_Temp', 'Top_Pressure', 'Reflux_Flow', 'Flow_To_Next_Process',
                                    'Temp_Sixth_Tray', 'Bottom_Temp_1', 'Bottom_Temp_2', 'C4_Fundo']).ffill()

# 3. Divisão estrita dos dados brutos antes de qualquer processamento matemático (Vazamento Zero)
split_limite = int(len(df_raw) * 0.8)
df_treino_bruto = df_raw.iloc[:split_limite].copy()
df_teste_bruto = df_raw.iloc[split_limite:].copy()

use_roll = (best['Rolling'] == True or best['Rolling'] == "True")
tamanho_janela_efetivo = int(best['Roll_Window']) if use_roll else 1

pipeline = IndustrialFeaturePipeline(
    span_ema=int(best['EMA_Span']),
    window_size=tamanho_janela_efetivo,
    lags_ccf=lags_dinamicos,
    lag_chromatograph=4
)

# 4. Aplicar EMA de forma isolada em cada bloco
df_treino_filt = pipeline.aplicar_filtro_ema(df_treino_bruto)
df_teste_filt = pipeline.aplicar_filtro_ema(df_teste_bruto)

# 5. Construir atributos de forma independente e isolada para cada bloco
def construir_atributos(df_pipeline, df_bruto):
    df_features = pd.DataFrame(index=df_pipeline.index)
    for col in pipeline.features_originais:
        sinal_filtrado = df_pipeline[col]
        lag_fisi = pipeline.lags_ccf[col]
        
        df_features[col] = sinal_filtrado.shift(lag_fisi)
        if best['Derivadas']:
            df_features[f"{col}_deriv"] = sinal_filtrado.diff().shift(lag_fisi)
        if best['Squared']:
            df_features[f"{col}_sq"] = (sinal_filtrado ** 2).shift(lag_fisi)
        if best['Rolling']:
            df_features[f"{col}_roll_mean"] = sinal_filtrado.rolling(window=tamanho_janela_efetivo).mean().shift(lag_fisi)
            df_features[f"{col}_roll_std"] = sinal_filtrado.rolling(window=tamanho_janela_efetivo).std().shift(lag_fisi)
            
    if 'C4_Fundo' in df_bruto.columns:
        df_features['C4_Fundo_lag_4'] = df_bruto['C4_Fundo'].shift(4)
        df_features['target'] = df_bruto['C4_Fundo']
        
    return df_features.dropna()

df_treino_final = construir_atributos(df_treino_filt, df_treino_bruto)
df_teste_final = construir_atributos(df_teste_filt, df_teste_bruto)

# Capturar índices temporais reais para o gráfico final de rastreamento
indices_teste = df_teste_final.index

# Separar em X e y
X_train = df_treino_final.drop(columns=['target']).values
y_train = df_treino_final['target'].values
X_test = df_teste_final.drop(columns=['target']).values
y_test_real = df_teste_final['target'].values

# 6. Escalonamento
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 7. Recalcular a Validação Cruzada Temporal (Apenas no conjunto de treino)
grid_comps = list(range(1, min(X_train_scaled.shape[1] + 1, 21)))
tscv = TimeSeriesSplit(n_splits=4)
rmse_medios_cv = []

for n_comp in grid_comps:
    fold_rmse = []
    for train_index, val_index in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_index], X_train_scaled[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        pls_cv = PLSRegression(n_components=n_comp)
        pls_cv.fit(X_tr, y_tr)
        preds_val = pls_cv.predict(X_val).flatten()
        fold_rmse.append(np.sqrt(mean_squared_error(y_val, preds_val)))
    rmse_medios_cv.append(np.mean(fold_rmse))

# 8. Treinamento e Predição Final
pls_final = PLSRegression(n_components=int(best['Best_C']))
pls_final.fit(X_train_scaled, y_train)
y_pred = pls_final.predict(X_test_scaled).flatten()

# 9. Diagnóstico Estatístico de Resíduos
residuos = y_test_real - y_pred
stat_jb, p_jb = stats.jarque_bera(residuos)

print("\nANÁLISE DE RESÍDUOS DO MODELO ESCOLHIDO:")
print(f"• R² Calculado no Teste Cego: {r2_score(y_test_real, y_pred):.4f}")
print(f"• Média dos Resíduos:         {np.mean(residuos):.6f}")
print(f"• Teste de Jarque-Bera:       Estatística={stat_jb:.2f} | p-valor={p_jb:.4f}")
if p_jb > 0.05:
    print("  ✔️ Resíduos seguem uma distribuição estritamente normal.")
else:
    print("  ⚠️ Resíduos NÃO são perfeitamente normais.")
print("="*75)

# ==============================================================================
# GERAÇÃO DOS GRÁFICOS DE VALIDAÇÃO E RASTREAMENTO
# ==============================================================================

# FIGURA 1: Matriz de Diagnóstico Original de Resíduos (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# [0, 0] Gráfico Real vs Predito
axes[0, 0].scatter(y_test_real, y_pred, color='navy', alpha=0.5, edgecolor='w')
axes[0, 0].plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
axes[0, 0].set_title(f"Real vs. Predito | Linha [{INDICE_MODELO}]")
axes[0, 0].set_xlabel("Valor Real (Cromatógrafo)")
axes[0, 0].set_ylabel("Predição do Soft-Sensor (PLS)")
axes[0, 0].grid(True, linestyle=':')

# [0, 1] Homocedasticidade (Resíduos vs Predito)
axes[0, 1].scatter(y_pred, residuos, color='crimson', alpha=0.5, edgecolor='w')
axes[0, 1].axhline(0, color='black', linestyle='--', lw=2)
axes[0, 1].set_title("Comportamento dos Erros (Resíduos vs. Predição)")
axes[0, 1].set_xlabel("Valores Preditos")
axes[0, 1].set_ylabel("Resíduos (Erro)")
axes[0, 1].grid(True, linestyle=':')

# [1, 0] Distribuição dos Erros (Histograma vs Densidades)
count, bins, ignored = axes[1, 0].hist(residuos, bins=30, density=True, alpha=0.6, color='seagreen', edgecolor='black')
mu, std = np.mean(residuos), np.std(residuos)
x_ajuste = np.linspace(bins.min(), bins.max(), 100)
axes[1, 0].plot(x_ajuste, stats.norm.pdf(x_ajuste, mu, std), color='darkred', lw=2, label='Normal Teórica')
pd.Series(residuos).plot(kind='kde', ax=axes[1, 0], color='blue', lw=1.5, label='KDE Real')
axes[1, 0].set_title("Distribuição dos Erros (Histograma vs Densidade)")
axes[1, 0].set_xlabel("Resíduos")
axes[1, 0].set_ylabel("Densidade")
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle=':')

# [1, 1] Gráfico Q-Q (Quantil-Quantil)
stats.probplot(residuos, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[0].set_markerfacecolor('purple')
axes[1, 1].get_lines()[0].set_alpha(0.5)
axes[1, 1].get_lines()[1].set_color('black')
axes[1, 1].set_title("Gráfico Q-Q de Probabilidade Normal")
axes[1, 1].set_xlabel("Quantis Teóricos")
axes[1, 1].set_ylabel("Quantis Ordenados dos Resíduos")
axes[1, 1].grid(True, linestyle=':')

plt.tight_layout()

# FIGURA 2: Curva de Validação Cruzada Temporal (Busca do Melhor C)
plt.figure(figsize=(10, 5))
plt.plot(grid_comps, rmse_medios_cv, marker='o', linestyle='-', color='teal', lw=2, label='RMSE CV Médio')
plt.axvline(int(best['Best_C']), color='darkorange', linestyle='--', lw=2, 
            label=f"Melhor Componente Escolhido (C = {int(best['Best_C'])})")
plt.title(f"Curva de Validação Cruzada Temporal | Pipeline Linha [{INDICE_MODELO}]")
plt.xlabel("Número de Componentes Latentes do PLS")
plt.ylabel("RMSE CV (Média dos Folds)")
plt.xticks(grid_comps)
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()

# FIGURA 3: Rastreamento Temporal dos Sinais (Real vs Predito na Amostra de Teste)
plt.figure(figsize=(16, 6))
plt.plot(indices_teste, y_test_real, label='Valor Real (Cromatógrafo)', color='black', lw=1.8, alpha=0.8)
plt.plot(indices_teste, y_pred, label='Predição do Soft-Sensor (PLS)', color='dodgerblue', lw=1.5, linestyle='--')
plt.title(f"Rastreamento Temporal de Sinais | Conjunto de Teste (R² = {r2_score(y_test_real, y_pred):.4f})")
plt.xlabel("Índice da Amostra (Tempo)")
plt.ylabel("Concentração de C4 no Fundo")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':')
plt.tight_layout()

# Renderizar todas as janelas gráficas criadas
plt.show()