import os
import sys
import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Inserir o diretório raiz do projeto no path para encontrar o módulo src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features import IndustrialFeaturePipeline

def calcular_lags_dinamicos(df, ema_span, max_lag=30):
    """
    Calcula a Função de Correlação Cruzada (FCC) para o span atual da EMA
    e determina dinamicamente o lag ótimo para cada feature preditora.
    """
    df_temp = df.copy()
    features_originais = [col for col in df_temp.columns if col != 'C4_Fundo']
    
    # Aplica suavização temporária com o span atual para alinhar com a dinâmica filtrada
    for col in features_originais:
        df_temp[col] = df_temp[col].ewm(span=ema_span, adjust=False).mean()
        
    lags_otimos = {}
    target_sinal = df_temp['C4_Fundo'].values
    
    for col in features_originais:
        feature_sinal = df_temp[col].values
        melhor_lag = 0
        maior_corr = -1.0
        
        # Varre a janela de busca de tempo morto do processo
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = abs(np.corrcoef(feature_sinal, target_sinal)[0, 1])
            else:
                corr = abs(np.corrcoef(feature_sinal[:-lag], target_sinal[lag:])[0, 1])
                
            if not np.isnan(corr) and corr > maior_corr:
                maior_corr = corr
                melhor_lag = lag
                
        lags_otimos[col] = melhor_lag
        
    return lags_otimos

def evaluate_pipeline(df_raw, ema_span, use_deriv, roll_window, use_sq, grid_comps):
    """
    Versão corrigida sem Data Leakage.
    Isola estritamente as transformações e o cálculo de lags no conjunto de treino.
    """
    # 1. Divisão estrita dos dados brutos antes de qualquer processamento
    split_limite = int(len(df_raw) * 0.8)
    df_treino_bruto = df_raw.iloc[:split_limite].copy()
    df_teste_bruto = df_raw.iloc[split_limite:].copy()

    # 2. Determinação dos lags dinâmicos utilizando apenas dados de treino (vazamento 1 corrigido)
    lags_dinamicos = calcular_lags_dinamicos(df_treino_bruto, ema_span)
    
    use_roll = (roll_window is not False)
    tamanho_janela_efetivo = roll_window if use_roll else 1
    
    pipeline = IndustrialFeaturePipeline(
        span_ema=ema_span, 
        window_size=tamanho_janela_efetivo, 
        lags_ccf=lags_dinamicos,
        lag_chromatograph=4
    )
    
    # 3. Suavização EMA de forma isolada para treino e teste (vazamento 2 corrigido)
    df_treino_filt = pipeline.aplicar_filtro_ema(df_treino_bruto)
    df_teste_filt = pipeline.aplicar_filtro_ema(df_teste_bruto)
    
    # Helper para construir atributos sem que haja vazamento na transição (vazamento 3 corrigido)
    def construir_atributos(df_pipeline, df_bruto):
        df_features = pd.DataFrame(index=df_pipeline.index)
        for col in pipeline.features_originais:
            sinal_filtrado = df_pipeline[col]
            lag_fisi = pipeline.lags_ccf[col]
            
            df_features[col] = sinal_filtrado.shift(lag_fisi)
            if use_deriv:
                df_features[f"{col}_deriv"] = sinal_filtrado.diff().shift(lag_fisi)
            if use_sq:
                df_features[f"{col}_sq"] = (sinal_filtrado ** 2).shift(lag_fisi)
            if use_roll:
                df_features[f"{col}_roll_mean"] = sinal_filtrado.rolling(window=tamanho_janela_efetivo).mean().shift(lag_fisi)
                df_features[f"{col}_roll_std"] = sinal_filtrado.rolling(window=tamanho_janela_efetivo).std().shift(lag_fisi)
                
        if 'C4_Fundo' in df_bruto.columns:
            df_features['C4_Fundo_lag_4'] = df_bruto['C4_Fundo'].shift(4)
            df_features['target'] = df_bruto['C4_Fundo']
            
        return df_features.dropna()

    # 4. Extração independente de matrizes de features
    df_treino_final = construir_atributos(df_treino_filt, df_treino_bruto)
    df_teste_final = construir_atributos(df_teste_filt, df_teste_bruto)
    
    if len(df_treino_final) < 50 or len(df_teste_final) < 10:
        return None

    # 5. Isolar os conjuntos de matrizes finais prontas
    X_train = df_treino_final.drop(columns=['target']).values
    y_train = df_treino_final['target'].values
    
    X_test = df_teste_final.drop(columns=['target']).values
    y_test = df_teste_final['target'].values

    # 6. Escalonamento
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # --------------------------------------------------------------------------
    # Validação Cruzada Temporal (Apenas no conjunto de treino)
    # --------------------------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=4)
    melhor_rmse_cv = float('inf')
    melhor_C = None

    for n_comp in grid_comps:
        if n_comp > X_train_scaled.shape[1]:
            continue

        fold_rmse = []
        for train_index, val_index in tscv.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_index], X_train_scaled[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            pls_cv = PLSRegression(n_components=n_comp)
            pls_cv.fit(X_tr, y_tr)
            preds_val = pls_cv.predict(X_val).flatten()
            fold_rmse.append(np.sqrt(mean_squared_error(y_val, preds_val)))

        avg_rmse_val = np.mean(fold_rmse)
        if avg_rmse_val < melhor_rmse_cv:
            melhor_rmse_cv = avg_rmse_val
            melhor_C = n_comp

    if melhor_C is None:
        return None

    # --------------------------------------------------------------------------
    # Avaliação Final no Teste Cego
    # --------------------------------------------------------------------------
    pls_final = PLSRegression(n_components=melhor_C)
    pls_final.fit(X_train_scaled, y_train)
    y_pred = pls_final.predict(X_test_scaled).flatten()

    rmse_teste = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_teste = r2_score(y_test, y_pred)

    residuos = y_test - y_pred
    stat_jb, p_jb = stats.jarque_bera(residuos)

    output_res = {
        'EMA_Span': ema_span,
        'Derivadas': use_deriv,
        'Rolling': use_roll,
        'Roll_Window': roll_window if use_roll else None,
        'Squared': use_sq,
        'N_Features': X_train_scaled.shape[1],
        'Best_C': melhor_C,
        'RMSE': rmse_teste,
        'R2': r2_teste,
        'Jarque_Bera_Stat': round(stat_jb, 4),
        'Jarque_Bera_p': round(p_jb, 6)
    }
    
    for col, lag_valor in lags_dinamicos.items():
        output_res[f'Lag_{col}'] = lag_valor

    return output_res

if __name__ == "__main__":
    caminho = "/home/hermann/projeto_soft_sensor/data/raw/debutanizer_data.txt"
    data = np.loadtxt(caminho, skiprows=5)
    df_raw = pd.DataFrame(data)
    df_raw.columns = ['Top_Temp', 'Top_Pressure', 'Reflux_Flow', 'Flow_To_Next_Process',
                      'Temp_Sixth_Tray', 'Bottom_Temp_1', 'Bottom_Temp_2', 'C4_Fundo']
    df_raw = df_raw.ffill()

    # Configuração da malha de parâmetros simplificada
    grid_ema_span = [3, 6, 9, 12, 15, 18, 21, 24, 27]
    grid_deriv = [True, False]
    grid_roll_window = [False, 4, 6, 8, 10, 12, 14, 16, 18]
    grid_sq = [True, False]
    
    grid_comps = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]

    resultados = []
    
    combinacoes = list(itertools.product(
        grid_ema_span, grid_deriv, grid_roll_window, grid_sq
    ))

    print(f"Iniciando varredura combinatória para {len(combinacoes)} configurações de pipeline...")

    for i, (ema, der, roll_w, sq) in enumerate(combinacoes):
        res = evaluate_pipeline(df_raw, ema, der, roll_w, sq, grid_comps)
        if res is not None:
            resultados.append(res)

        if (i + 1) % 50 == 0:
            print(f"Progresso: {i + 1} / {len(combinacoes)} pipelines avaliados...")

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(by='R2', ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print("TOP 10 MELHORES CONFIGURAÇÕES DE PIPELINE (ORDENADO POR R²):")
    print("="*80)
    print(df_resultados.head(10).to_string(index=True))
    print("="*80 + "\n")

    os.makedirs("data/processed", exist_ok=True)
    caminho_salvar = "data/processed/resultado_otimizacao_pls.csv"
    df_resultados.to_csv(caminho_salvar, index=False)
    print(f"Tabela completa de resultados salva em: {caminho_salvar}")