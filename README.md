## Como obter os dados

Os dados utilizados neste projeto são históricos reais de uma coluna debutanizadora industrial provenientes do livro 'Soft Sensors for Monitoring and Control of Industrial Processes' (Fortuna et al., 2007). Devido a boas práticas de armazenamento, os dados originais não estão incluídos neste repositório, mas podem ser baixados em [(https://extras.springer.com/downloads/sgw-extras/2007/978-1-84628-480-9)]. Para rodar o projeto, baixe o arquivo "debutanizer_data.txt" e insira-o na pasta data/raw/

# Desenvolvimento de Soft-Sensor Industrial para Coluna de Debutanização via MW-PLS

## 1. Introdução e Contexto Físico
- Descrição da coluna de debutanização (remoção de C4 da corrente de gasolina).
- O gargalo: Atraso de 40 minutos (tempo morto de análise + tempo de ciclo do cromatógrafo).
- A solução: Estimador de composição em tempo real (Soft-Sensor) usando variáveis térmicas e hidráulicas de resposta rápida.

## 2. Metodologia de Pré-processamento Causal (Vazamento Zero)
- Alinhamento dinâmico de lags físicos por correlação cruzada (FCC) com restrição física.
- Filtragem exponencial (EMA) isolada por bloco.
- Detalhamento de como o Data Leakage foi evitado na transição de treino-teste e nas rolling features.

## 3. Otimização Estática vs. Dinâmica
- Comparação das tabelas de hiperparâmetros.
- Discussão sobre a parcimônia (Razão C/F, componentes latentes e interpretabilidade física).

## 4. Resultados e Diagnósticos de Processo
- Exibição dos gráficos de Rastreamento de Sinais e Homocedasticidade.
- Discussão física sobre o teste de Jarque-Bera nos resíduos.
- Análise da evolução temporal dos coeficientes do MW-PLS para comprovar a adaptação a perturbações dinâmicas de longo termo.