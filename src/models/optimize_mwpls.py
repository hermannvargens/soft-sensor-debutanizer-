import os
import sys
import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Inserir o diretório raiz do projeto no path para encontrar o módulo src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features import IndustrialFeaturePipeline

def calcular_lags_dinamicos(df, ema_span, max_lag=30):
    """Calcula a correlação cruzada sob a filtragem EMA atual."""
    df_temp = df.copy()
    features_originais = [col for col in df_temp.columns if col != 'C4_Fundo']
    for col in features_originais:
        df_temp[col] = df_temp[col].ewm(span=ema_span, adjust=False).mean()
        
    lags_otimos = {}
    target_sinal = df_temp['C4_Fundo'].values
    for col in features_originais:
        feature_sinal = df_temp[col].values
        melhor_lag, maior_corr = 0, -1.0
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

def simular_mwpls(df_pipeline, df_teste_final, window_size_mw, n_components):
    """
    Simula o comportamento online do MW-PLS passo a passo.
    Garante o isolamento estatístico absoluto e alinhamento temporal rigoroso.
    """
    X = df_pipeline.drop(columns=['target']).values
    y = df_pipeline['target'].values
    
    # Determina a posição física exata onde o teste real começa no pipeline (Alinhamento Rigoroso)
    primeiro_indice_teste = df_teste_final.index[0]
    split_idx = df_pipeline.index.get_loc(primeiro_indice_teste)
    
    y_reais_teste = []
    y_preds_teste = []
    
    # Varredura temporal passo a passo sobre o conjunto de teste real
    for t in range(split_idx, len(df_pipeline)):
        # Janela de calibração local imediatamente anterior ao instante t
        idx_inicio_janela = t - window_size_mw
        
        # Proteção para garantir que temos histórico suficiente para preencher a janela móvel
        if idx_inicio_janela < 0:
            continue
            
        X_calib = X[idx_inicio_janela:t]
        y_calib = y[idx_inicio_janela:t]
        
        X_atual = X[t].reshape(1, -1)
        
        # Escalonamento local e isolado (Evita data leakage global)
        scaler_X_local = StandardScaler()
        X_calib_scaled = scaler_X_local.fit_transform(X_calib)
        X_atual_scaled = scaler_X_local.transform(X_atual)
        
        # Treinamento do modelo dinâmico na janela atual
        try:
            model_mw = PLSRegression(n_components=n_components)
            model_mw.fit(X_calib_scaled, y_calib)
            
            pred_atual = model_mw.predict(X_atual_scaled).flatten()[0]
            
            y_reais_teste.append(y[t])
            y_preds_teste.append(pred_atual)
        except Exception:
            # Protege contra instabilidades numéricas se n_components for inadequado para a janela
            continue
            
    if len(y_preds_teste) < 50:
        return None, None, None, None
        
    y_reais_teste = np.array(y_reais_teste)
    y_preds_teste = np.array(y_preds_teste)
    
    rmse = np.sqrt(mean_squared_error(y_reais_teste, y_preds_teste))
    r2 = r2_score(y_reais_teste, y_preds_teste)
    
    # Análise de resíduos do simulado
    residuos = y_reais_teste - y_preds_teste
    stat_jb, p_jb = stats.jarque_bera(residuos)
    
    return rmse, r2, stat_jb, p_jb

def evaluate_mwpls_pipeline(df_raw, ema_span, use_deriv, roll_window, use_sq, window_size_mw, n_components):
    """
    Versão com isolamento temporal absoluto para evitar vazamentos na EMA e Rolling Features.
    """
    # 1. Divisão estrita de dados brutos antes de qualquer processamento matemático
    split_limite = int(len(df_raw) * 0.8)
    df_treino_bruto = df_raw.iloc[:split_limite].copy()
    df_teste_bruto = df_raw.iloc[split_limite:].copy()

    # 2. Lags dinâmicos calculados estritamente na partição de treino
    lags_dinamicos = calcular_lags_dinamicos(df_treino_bruto, ema_span)

    use_roll = (roll_window is not False)
    tamanho_janela_efetivo = roll_window if use_roll else 1
    
    pipeline = IndustrialFeaturePipeline(
        span_ema=ema_span, 
        window_size=tamanho_janela_efetivo, 
        lags_ccf=lags_dinamicos,
        lag_chromatograph=4
    )
    
    # 3. Aplicar EMA de forma isolada em cada bloco (Sem vazamento de fronteira)
    df_treino_filt = pipeline.aplicar_filtro_ema(df_treino_bruto)
    df_teste_filt = pipeline.aplicar_filtro_ema(df_teste_bruto)
    
    # 4. Construir atributos para treino e teste de forma independente
    def extrair_atributos_bloco(df_pipeline, df_bruto_alvo):
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
                
        if 'C4_Fundo' in df_bruto_alvo.columns:
            df_features['C4_Fundo_lag_4'] = df_bruto_alvo['C4_Fundo'].shift(4)
            df_features['target'] = df_bruto_alvo['C4_Fundo']
        return df_features

    df_features_treino = extrair_atributos_bloco(df_treino_filt, df_treino_bruto).dropna()
    df_features_teste = extrair_atributos_bloco(df_teste_filt, df_teste_bruto).dropna()
    
    # 5. Concatenar as matrizes de atributos limpas para a simulação do MW-PLS
    # Agora a transição de índices está protegida contra NaNs e contaminações
    df_pipeline = pd.concat([df_features_treino, df_features_teste])
    
    # 6. Avalia a simulação causal e dinâmica do MW-PLS passando a matriz de teste isolada
    rmse, r2, stat_jb, p_jb = simular_mwpls(df_pipeline, df_features_teste, window_size_mw, n_components)
    
    if rmse is None:
        return None
        
    output_res = {
        'EMA_Span': ema_span,
        'Derivadas': use_deriv,
        'Rolling': use_roll,
        'Roll_Window': roll_window if use_roll else None,
        'Squared': use_sq,
        'MW_Window': window_size_mw,
        'N_Features': df_pipeline.shape[1] - 1,
        'Components_C': n_components,
        'RMSE': rmse,
        'R2': r2,
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

    # Espaço de Busca Otimizado (Focado na viabilidade computacional)
    grid_ema_span = [6, 9, 12, 15, 18, 21]
    grid_deriv = [True, False]
    grid_roll_window = [False, 4, 8, 12, 16, 18]
    grid_sq = [True, False]
    
    # Novos Hiperparâmetros Dinâmicos do MW-PLS
    grid_mw_window = [200, 300, 400] # Tamanho da memória do refervedor
    grid_comps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    resultados = []
    
    combinacoes = list(itertools.product(
        grid_ema_span, grid_deriv, grid_roll_window, grid_sq, grid_mw_window, grid_comps
    ))

    print(f"Iniciando varredura dinâmica para {len(combinacoes)} configurações de MW-PLS...")

    for i, (ema, der, roll_w, sq, mw_w, n_c) in enumerate(combinacoes):
        # Validação: n_components locais não podem ser maiores que as features geradas estimadas
        estimativa_features = 8
        if der: estimativa_features += 7
        if sq: estimativa_features += 7
        if roll_w is not False: estimativa_features += 14
        
        if n_c > estimativa_features:
            continue

        res = evaluate_mwpls_pipeline(df_raw, ema, der, roll_w, sq, mw_w, n_c)
        if res is not None:
            resultados.append(res)

        if (i + 1) % 50 == 0:
            print(f"Progresso: {i + 1} / {len(combinacoes)} pipelines dinâmicos avaliados...")

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(by='R2', ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print("TOP 10 MELHORES CONFIGURAÇÕES DE MW-PLS:")
    print("="*80)
    print(df_resultados.head(10).to_string(index=True))
    print("="*80 + "\n")

    os.makedirs("data/processed", exist_ok=True)
    caminho_salvar = "data/processed/resultado_otimizacao_mwpls.csv"
    df_resultados.to_csv(caminho_salvar, index=False)
    print(f"Tabela completa de resultados de MW-PLS salva em: {caminho_salvar}")