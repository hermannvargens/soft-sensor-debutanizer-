import os
import sys
import numpy as np
import pandas as pd
import joblib

# Inserir o diretório raiz do projeto no path para encontrar o módulo src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features import IndustrialFeaturePipeline

def treinar_e_salvar_mwpls():
    print("="*75)
    print("PREPARANDO EXPORTAÇÃO DOS ARTEFATOS MW-PLS (PRODUÇÃO)")
    print("="*75)
    
    # 1. Carregar os resultados da otimização do MW-PLS
    caminho_csv = "data/processed/resultado_otimizacao_mwpls.csv"
    if not os.path.exists(caminho_csv):
        print("Erro: O arquivo de otimização do MW-PLS não foi encontrado.")
        sys.exit(1)
        
    df_res = pd.read_csv(caminho_csv)
    
    # Selecionar o modelo campeão (Linha 0 do ranking de R²)
    INDICE = 0
    best = df_res.iloc[INDICE]
    
    # Extração dinâmica dos lags do CSV
    colunas_lags = [col for col in df_res.columns if col.startswith('Lag_')]
    lags_dinamicos = {col.replace('Lag_', ''): int(best[col]) for col in colunas_lags}
    
    print(f"• Configuração Escolhida: EMA_Span={int(best['EMA_Span'])} | Derivadas={best['Derivadas']} | "
          f"Rolling={best['Rolling']} | Janela={int(best['Roll_Window']) if best['Rolling'] else 'N/A'}")
    print(f"• Memória MW: {int(best['MW_Window'])} amostras | Componentes C: {int(best['Components_C'])}")
    print(f"• R² Esperado: {best['R2']:.4f}")
    print("-"*75)

    # 2. Carregar dados históricos da planta
    caminho_dados = "/home/hermann/projeto_soft_sensor/data/raw/debutanizer_data.txt"
    data = np.loadtxt(caminho_dados, skiprows=5)
    df_raw = pd.DataFrame(data, columns=['Top_Temp', 'Top_Pressure', 'Reflux_Flow', 'Flow_To_Next_Process',
                                        'Temp_Sixth_Tray', 'Bottom_Temp_1', 'Bottom_Temp_2', 'C4_Fundo']).ffill()

    # 3. Construir o pipeline com os parâmetros do campeão
    use_roll = (best['Rolling'] == True or best['Rolling'] == "True")
    tamanho_janela_efetivo = int(best['Roll_Window']) if use_roll else 1

    pipeline = IndustrialFeaturePipeline(
        span_ema=int(best['EMA_Span']),
        window_size=tamanho_janela_efetivo,
        lags_ccf=lags_dinamicos,
        lag_chromatograph=4
    )
    
    # 4. Processar o histórico final de dados completo
    print("Processando matriz de atributos históricos...")
    df_filt = pipeline.aplicar_filtro_ema(df_raw)
    
    df_features = pd.DataFrame(index=df_filt.index)
    for col in pipeline.features_originais:
        sinal_filtrado = df_filt[col]
        lag_fisi = pipeline.lags_ccf[col]
        
        df_features[col] = sinal_filtrado.shift(lag_fisi)
        if best['Derivadas']:
            df_features[f"{col}_deriv"] = sinal_filtrado.diff().shift(lag_fisi)
        if best['Squared']:
            df_features[f"{col}_sq"] = (sinal_filtrado ** 2).shift(lag_fisi)
        if best['Rolling']:
            df_features[f"{col}_roll_mean"] = sinal_filtrado.rolling(window=tamanho_janela_efetivo).mean().shift(lag_fisi)
            df_features[f"{col}_roll_std"] = sinal_filtrado.rolling(window=tamanho_janela_efetivo).std().shift(lag_fisi)
            
    df_features['C4_Fundo_lag_4'] = df_raw['C4_Fundo'].shift(4)
    df_features['target'] = df_raw['C4_Fundo']
    df_pipeline_completo = df_features.dropna()
    
    # 5. Isolar o buffer de partida online
    # Em produção, o script precisará de um histórico de tamanho W imediatamente disponível
    # para preencher a janela deslizante e conseguir predizer o ponto atual sem esperar
    tamanho_buffer = int(best['MW_Window'])
    df_buffer_partida = df_pipeline_completo.iloc[-tamanho_buffer:]
    
    print(f"Criando buffer inicial de produção com as últimas {tamanho_buffer} amostras limpas...")
    
    # 6. Salvar metadados e objetos em disco
    os.makedirs("models", exist_ok=True)
    
    # Dicionário com os hiperparâmetros dinâmicos de controle
    config_mwpls = {
        'window_size_mw': int(best['MW_Window']),
        'n_components': int(best['Components_C']),
        'features_names': list(df_pipeline_completo.drop(columns=['target']).columns),
        'best_row_index': INDICE,
        'pipeline_params': {
            'span_ema': int(best['EMA_Span']),
            'window_size': tamanho_janela_efetivo,
            'lags_ccf': lags_dinamicos,
            'lag_chromatograph': 4,
            'use_deriv': best['Derivadas'],
            'use_sq': best['Squared'],
            'use_roll': use_roll
        }
    }
    
    joblib.dump(config_mwpls, "models/mwpls_config.joblib")
    joblib.dump(df_buffer_partida, "models/mwpls_buffer_partida.joblib")
    
    print("Artefatos dinâmicos exportados com sucesso para /models!")
    print("  - mwpls_config.joblib (Configurações e Hiperparâmetros)")
    print("  - mwpls_buffer_partida.joblib (Dados de calibração online iniciais)")
    print("="*75)

if __name__ == "__main__":
    treinar_e_salvar_mwpls()