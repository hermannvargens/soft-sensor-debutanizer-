import pandas as pd
import numpy as np

class IndustrialFeaturePipeline:
    def __init__(self, span_ema=18, window_size=14, lags_ccf=None, lag_chromatograph=4):
        """
        Pipeline flexível de Engenharia de Atributos com suporte a Rolling Features.
        """
        self.span_ema = span_ema
        self.window_size = window_size
        self.lag_chromatograph = lag_chromatograph
        
        # Se o usuário não passar os lags, usamos o padrão calibrado na EDA para o span=18
        if lags_ccf is None:
            self.lags_ccf = {
                'Top_Temp': 11,
                'Top_Pressure': 0,
                'Reflux_Flow': 3,
                'Flow_To_Next_Process': 0,
                'Temp_Sixth_Tray': 9,
                'Bottom_Temp_1': 11,
                'Bottom_Temp_2': 11
            }
        else:
            self.lags_ccf = lags_ccf
            
        self.features_originais = list(self.lags_ccf.keys())

    def aplicar_filtro_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a Média Móvel Exponencial (Causal) para atenuação de ruído."""
        df_filtrado = df.copy()
        for col in self.features_originais:
            if col in df_filtrado.columns:
                df_filtrado[col] = df_filtrado[col].ewm(span=self.span_ema, adjust=False).mean()
        return df_filtrado

    def construir_atributos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa o pipeline completo: Filtro, Rolling Features, Lags, Derivadas e Quadráticos.
        """
        # 1. Filtragem por EMA
        df_pipeline = self.aplicar_filtro_ema(df)
        
        df_features = pd.DataFrame(index=df_pipeline.index)
        
        # 2. Processamento de cada variável física
        for col in self.features_originais:
            sinal_filtrado = df_pipeline[col]
            
            # A. Engenharia de Atributos instantâneos e diferenciais
            sinal_deriv = sinal_filtrado.diff()
            sinal_sq = sinal_filtrado ** 2
            
            # B. Extração de Recursos via Janelas Deslizantes (Rolling Features) antes do shift
            sinal_roll_mean = sinal_filtrado.rolling(window=self.window_size).mean()
            sinal_roll_std = sinal_filtrado.rolling(window=self.window_size).std()
            
            # C. Aplicar o shift do Lag Ótimo individual da variável física e suas transformações
            lag_fisi = self.lags_ccf[col]
            
            df_features[f"{col}"] = sinal_filtrado.shift(lag_fisi)
            df_features[f"{col}_deriv"] = sinal_deriv.shift(lag_fisi)
            df_features[f"{col}_sq"] = sinal_sq.shift(lag_fisi)
            df_features[f"{col}_roll_mean"] = sinal_roll_mean.shift(lag_fisi)
            df_features[f"{col}_roll_std"] = sinal_roll_std.shift(lag_fisi)
            
        # 3. Tratamento da Variável Alvo (Target Autorregressiva)
        if 'C4_Fundo' in df.columns:
            # Shift de 4 passos para simular o atraso real de análise do cromatógrafo
            df_features['C4_Fundo_lag_4'] = df['C4_Fundo'].shift(self.lag_chromatograph)
            df_features['target'] = df['C4_Fundo'] # Alvo atual para o treinamento
            
        # 4. Saneamento: Limpar os NaNs gerados pelos Lags, Derivadas e Janelas Móveis
        df_features = df_features.dropna()
        
        return df_features

if __name__ == "__main__":
    print("="*60)
    print(" 🚀 INICIANDO TESTE DO PIPELINE DE RECURSOS INDUSTRIAIS")
    print("="*60)
    
    # 1. Carregamento com tratamento de índice dos dados reais
    caminho = "/home/hermann/projeto_soft_sensor/data/raw/debutanizer_data.txt"
    data = np.loadtxt(caminho, skiprows=5)
    df = pd.DataFrame(data)

    # 2. Renomeação correta seguindo a ordem física do arquivo original
    df.columns = [
        'Top_Temp',             # x1
        'Top_Pressure',         # x2
        'Reflux_Flow',          # x3
        'Flow_To_Next_Process', # x4
        'Temp_Sixth_Tray',      # x5
        'Bottom_Temp_1',        # x6
        'Bottom_Temp_2',        # x7
        'C4_Fundo',             # y
    ]
    
    df_raw_teste = df.copy()
    print(f"✔️ Dataset bruto real carregado com sucesso! Formato original: {df_raw_teste.shape}")
    
    # 3. Instanciar o nosso Pipeline configurando a janela deslizante para 10 passos (120 minutos)
    pipeline = IndustrialFeaturePipeline(span_ema=18, window_size=10, lag_chromatograph=4)
    
    # 4. Executar a transformação completa
    print("⚙️ Processando filtros, rolling features, lags ótimos, derivadas e termos quadráticos...")
    df_resultado = pipeline.construir_atributos(df_raw_teste)
    
    # 5. Mostrar os resultados na tela para validação visual
    print("="*60)
    print("📊 RESULTADOS DO PIPELINE DE PRODUÇÃO:")
    print("="*60)
    print(f"• Formato final da matriz de treino: {df_resultado.shape}")
    print(f"  (Repare que o número de linhas reduziu devido ao dropna acumulado das janelas e lags)")
    
    # Validação de que as novas variáveis móveis foram criadas
    colunas_checagem = [
        'Temp_Sixth_Tray', 
        'Temp_Sixth_Tray_roll_mean', 
        'Temp_Sixth_Tray_roll_std', 
        'C4_Fundo_lag_4', 
        'target'
    ]
    print("\n• Amostra das novas Rolling Features geradas para a 6ª Bandeja:")
    print(df_resultado[colunas_checagem].head(3))
    print("="*60)
    print("🎉 TESTE CONCLUÍDO COM SUCESSO! PIPELINE PRONTO PARA O PLS ESTÁTICO.")
    print("="*60)