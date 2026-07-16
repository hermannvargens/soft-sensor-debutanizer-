import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features import IndustrialFeaturePipeline

# ==============================================================================
# CONFIGURAÇÃO DE DIAGNÓSTICO: ESCOLHA A LINHA DO CSV DE MW-PLS QUE DESEJA AVALIAR
# ==============================================================================
INDICE_MODELO = 456  # Ajuste para a linha do CSV de otimização do MW-PLS

def salvar_predicoes_teste(indices, y_real, y_pred, caminho_saida="data/processed/predicoes_teste_mwpls.csv"):
    """
    Salva os resultados da simulação temporal do teste em um arquivo CSV 
    para permitir comparações externas com os dados brutos.
    """
    df_export = pd.DataFrame({
        'C4_Fundo_Real': y_real,
        'C4_Fundo_Pred_MWPLS': y_pred
    }, index=indices)
    
    # Garante que a pasta de destino exista
    diretorio = os.path.dirname(caminho_saida)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)
        
    df_export.to_csv(caminho_saida, index_label='Indice_Temporal')
    print(f"Dados de teste exportados com sucesso para: {caminho_saida}")


# 1. Carregar os resultados da otimização do MW-PLS
caminho_csv = "data/processed/resultado_otimizacao_mwpls.csv"
if not os.path.exists(caminho_csv):
    print(f"Erro: O arquivo {caminho_csv} não foi encontrado. Rode optimize_mwpls.py primeiro.")
    sys.exit(1)

df_res = pd.read_csv(caminho_csv)

if INDICE_MODELO >= len(df_res):
    print(f"Erro: O índice {INDICE_MODELO} está fora dos limites (máximo: {len(df_res)-1}).")
    sys.exit(1)

best = df_res.iloc[INDICE_MODELO]

# Extração dos lags salvos
colunas_lags = [col for col in df_res.columns if col.startswith('Lag_')]
lags_dinamicos = {col.replace('Lag_', ''): int(best[col]) for col in colunas_lags}

print("="*75)
print(f"AVALIANDO MW-PLS - LINHA [{INDICE_MODELO}] DO CSV DE OTIMIZAÇÃO")
print("="*75)
print(f"• Configuração: EMA_Span={int(best['EMA_Span'])} | Derivadas={best['Derivadas']} | "
      f"Rolling={best['Rolling']} | Janela={int(best['Roll_Window']) if best['Rolling'] else 'N/A'} | "
      f"Quadráticos={best['Squared']}")
print(f"• Memória MW:   {int(best['MW_Window'])} amostras | {int(best['Components_C'])} Componentes locais")
print(f"• R² do Grid:    {best['R2']:.4f} | RMSE: {best['RMSE']:.6f}")
print("="*75)

# 2. Carregar dados reais da planta
caminho_dados = "/home/hermann/projeto_soft_sensor/data/raw/debutanizer_data.txt"
data = np.loadtxt(caminho_dados, skiprows=5)
df_raw = pd.DataFrame(data, columns=['Top_Temp', 'Top_Pressure', 'Reflux_Flow', 'Flow_To_Next_Process',
                                    'Temp_Sixth_Tray', 'Bottom_Temp_1', 'Bottom_Temp_2', 'C4_Fundo']).ffill()

# 3. Divisão estrita de treino/teste (Vazamento Zero)
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

# 4. Processamento da EMA isolado
df_treino_filt = pipeline.aplicar_filtro_ema(df_treino_bruto)
df_teste_filt = pipeline.aplicar_filtro_ema(df_teste_bruto)

# 5. Construção de atributos isolada
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

df_pipeline = pd.concat([df_treino_final, df_teste_final])

# ==============================================================================
# 6. Simulação do MW-PLS (CORRIGIDA: Alinhamento temporal rigoroso)
# ==============================================================================
X = df_pipeline.drop(columns=['target']).values
y = df_pipeline['target'].values
nomes_features = df_pipeline.drop(columns=['target']).columns

# Em vez de calcular 80% do dataframe encolhido, localizamos a posição física exata 
# onde o conjunto de teste real (df_teste_final) começa dentro do df_pipeline
primeiro_indice_teste = df_teste_final.index[0]
split_idx = df_pipeline.index.get_loc(primeiro_indice_teste)

window_size_mw = int(best['MW_Window'])
n_components = int(best['Components_C'])

y_reais_teste = []
y_preds_teste = []
indices_teste = []
historico_coeficientes = []

# O loop agora começa RIGOROSAMENTE no primeiro ponto do conjunto de teste (ex: 1950)
for t in range(split_idx, len(df_pipeline)):
    idx_inicio = t - window_size_mw
    
    if idx_inicio < 0:
        continue
        
    X_calib = X[idx_inicio:t]
    y_calib = y[idx_inicio:t]
    X_atual = X[t].reshape(1, -1)
    
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib)
    X_atual_scaled = scaler.transform(X_atual)
    
    try:
        model_mw = PLSRegression(n_components=n_components)
        model_mw.fit(X_calib_scaled, y_calib)
        
        pred = model_mw.predict(X_atual_scaled).flatten()[0]
        
        y_preds_teste.append(pred)
        y_reais_teste.append(y[t])
        indices_teste.append(df_pipeline.index[t])
        
        historico_coeficientes.append(model_mw.coef_.flatten())
    except Exception:
        continue

y_reais_teste = np.array(y_reais_teste)
y_preds_teste = np.array(y_preds_teste)
residuos = y_reais_teste - y_preds_teste
stat_jb, p_jb = stats.jarque_bera(residuos)

df_coefs = pd.DataFrame(historico_coeficientes, columns=nomes_features, index=indices_teste)

print("\nANÁLISE DE RESÍDUOS DO MW-PLS:")
print(f"• R² Real no Teste Cego:  {r2_score(y_reais_teste, y_preds_teste):.4f}")
print(f"• RMSE no Teste Cego:    {np.sqrt(mean_squared_error(y_reais_teste, y_preds_teste)):.6f}")
print(f"• Média dos Resíduos:     {np.mean(residuos):.6f}")
print(f"• Teste de Jarque-Bera:   Estatística={stat_jb:.2f} | p-valor={p_jb:.4f}")
if p_jb > 0.05:
    print("  ✔️ Resíduos seguem uma distribuição estritamente normal.")
else:
    print("  ⚠️ Resíduos não são perfeitamente normais.")
print("="*75)

# ==============================================================================
# CHAMADA DA FUNÇÃO DE EXPORTAÇÃO DOS DADOS
# ==============================================================================
salvar_predicoes_teste(indices_teste, y_reais_teste, y_preds_teste)

# ==============================================================================
# PLOTS DE DIAGNÓSTICO E ADAPTAÇÃO TEMPORAL
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# [0, 0] Real vs Predito
axes[0, 0].scatter(y_reais_teste, y_preds_teste, color='navy', alpha=0.5, edgecolor='w')
axes[0, 0].plot([y_reais_teste.min(), y_reais_teste.max()], [y_reais_teste.min(), y_reais_teste.max()], 'r--', lw=2)
axes[0, 0].set_title(f"MW-PLS: Real vs. Predito | Linha [{INDICE_MODELO}]")
axes[0, 0].set_xlabel("Valor Real (Cromatógrafo)")
axes[0, 0].set_ylabel("Predição Online (MW-PLS)")
axes[0, 0].grid(True, linestyle=':')

# [0, 1] Erro vs Predição (Homocedasticidade)
axes[0, 1].scatter(y_preds_teste, residuos, color='crimson', alpha=0.5, edgecolor='w')
axes[0, 1].axhline(0, color='black', linestyle='--', lw=2)
axes[0, 1].set_title("Comportamento dos Erros (Resíduos vs. Predição)")
axes[0, 1].set_xlabel("Valores Preditos")
axes[0, 1].set_ylabel("Resíduos (Erro)")
axes[0, 1].grid(True, linestyle=':')

# [1, 0] Distribuição dos Erros
count, bins, ignored = axes[1, 0].hist(residuos, bins=30, density=True, alpha=0.6, color='seagreen', edgecolor='black')
mu, std = np.mean(residuos), np.std(residuos)
x_ajuste = np.linspace(bins.min(), bins.max(), 100)
axes[1, 0].plot(x_ajuste, stats.norm.pdf(x_ajuste, mu, std), color='darkred', lw=2, label='Normal Teórica')
pd.Series(residuos).plot(kind='kde', ax=axes[1, 0], color='blue', lw=1.5, label='KDE Real')
axes[1, 0].set_title("Distribuição dos Erros do MW-PLS")
axes[1, 0].set_xlabel("Resíduos")
axes[1, 0].set_ylabel("Densidade")
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle=':')

# [1, 1] Q-Q Plot
stats.probplot(residuos, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[0].set_markerfacecolor('purple')
axes[1, 1].get_lines()[0].set_alpha(0.5)
axes[1, 1].get_lines()[1].set_color('black')
axes[1, 1].set_title("Gráfico Q-Q do MW-PLS")
axes[1, 1].set_xlabel("Quantis Teóricos")
axes[1, 1].set_ylabel("Quantis Ordenados")
axes[1, 1].grid(True, linestyle=':')

plt.tight_layout()

# ==============================================================================
# FIGURA 2: Rastreamento Temporal dos Sinais
# ==============================================================================
# Pegamos o primeiro índice do teste bruto (1915/1916) para mostrar todo o histórico real do teste
indice_inicio_teste_real = df_teste_bruto.index[0]
df_real_teste_completo = df_raw['C4_Fundo'].loc[indice_inicio_teste_real:]

plt.figure(figsize=(16, 6))

# 1. Mostra a curva real medida completa do teste (começando em 1915/1916)
plt.plot(df_real_teste_completo.index, df_real_teste_completo.values, 
         label='Valor Real (Cromatógrafo)', color='black', lw=1.8, alpha=0.8)

# 2. Mostra a predição apenas onde ela existe (começando em 1950)
plt.plot(indices_teste, y_preds_teste, 
         label='Predição Adaptativa (MW-PLS)', color='forestgreen', lw=1.5, linestyle='--')

plt.title(f"MW-PLS Rastreamento Temporal de Sinais | R² = {r2_score(y_reais_teste, y_preds_teste):.4f}")
plt.xlabel("Índice Temporal")
plt.ylabel("Concentração de C4 no Fundo")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':')
plt.tight_layout()

# FIGURA 3: Evolução dos Coeficientes do Modelo
plt.figure(figsize=(16, 6))
variaveis_plot = [col for col in pipeline.features_originais]
for col in variaveis_plot:
    plt.plot(indices_teste, df_coefs[col], label=f'β_{col}', lw=1.5)
plt.title("Evolução Temporal dos Coeficientes PLS (Adaptação a Drift de Processo)")
plt.xlabel("Índice Temporal")
plt.ylabel("Magnitude do Coeficiente (Normalizado)")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle=':')
plt.tight_layout()

plt.show()