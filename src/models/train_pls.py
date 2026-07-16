import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features import IndustrialFeaturePipeline

def treinar_e_salvar_melhor_modelo():
    print("="*75)
    print("PREPARANDO EXPORTAÇÃO DO MELHOR MODELO PLS ESTÁTICO")
    print("="*75)
    
    # 1. Ler o CSV de otimização e pegar a linha selecionada (ex: linha 119 ou 138)
    caminho_csv = "data/processed/resultado_otimizacao_pls.csv"
    if not os.path.exists(caminho_csv):
        print("❌ Erro: Execute o optimize_pls.py primeiro.")
        return
        
    df_res = pd.read_csv(caminho_csv)
    
    # INDICE DO MODELO ESCOLHIDO (Altere para o índice do modelo campeão, ex: 119 ou 138)
    INDICE = 105 
    best = df_res.iloc[INDICE]
    
    # Extração dinâmica de lags
    colunas_lags = [col for col in df_res.columns if col.startswith('Lag_')]
    lags_dinamicos = {col.replace('Lag_', ''): int(best[col]) for col in colunas_lags}
    
    # 2. Carregar os dados reais
    caminho_dados = "/home/hermann/projeto_soft_sensor/data/raw/debutanizer_data.txt"
    data = np.loadtxt(caminho_dados, skiprows=5)
    df_raw = pd.DataFrame(data, columns=['Top_Temp', 'Top_Pressure', 'Reflux_Flow', 'Flow_To_Next_Process',
                                        'Temp_Sixth_Tray', 'Bottom_Temp_1', 'Bottom_Temp_2', 'C4_Fundo']).ffill()

    # 3. Processar atributos
    pipeline = IndustrialFeaturePipeline(
        span_ema=int(best['EMA_Span']),
        window_size=int(best['Roll_Window']) if best['Rolling'] else 10,
        lags_ccf=lags_dinamicos
    )
    
    df_pipeline = pipeline.aplicar_filtro_ema(df_raw)
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
            df_features[f"{col}_roll_mean"] = sinal_filtrado.rolling(window=pipeline.window_size).mean().shift(lag_fisi)
            df_features[f"{col}_roll_std"] = sinal_filtrado.rolling(window=pipeline.window_size).std().shift(lag_fisi)
            
    df_features['C4_Fundo_lag_4'] = df_raw['C4_Fundo'].shift(4)
    df_features['target'] = df_raw['C4_Fundo']
    df_pipeline = df_features.dropna()
    
    # 4. Preparar matrizes de Treino Completo (ou divisão 80/20)
    X = df_pipeline.drop(columns=['target']).values
    y = df_pipeline['target'].values
    
    # Padronização
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 5. Treinar o modelo final
    print(f"Treinando PLSRegression final com {int(best['Best_C'])} componentes...")
    model_pls = PLSRegression(n_components=int(best['Best_C']))
    model_pls.fit(X_scaled, y)
    
    # 6. Salvar artefatos estruturados em disco
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model_pls, "models/pls_static_model.joblib")
    joblib.dump(scaler_X, "models/scaler_x.joblib")
    joblib.dump(pipeline, "models/pipeline_config.joblib")
    
    print("Artefatos exportados com sucesso para a pasta /models!")
    print("  - pls_static_model.joblib (Pesos do modelo)")
    print("  - scaler_x.joblib (Média e variância de treino)")
    print("  - pipeline_config.joblib (Classe de variáveis configurada)")
    print("="*75)

if __name__ == "__main__":
    treinar_e_salvar_melhor_modelo()