import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Defina o período de tempo desejado
data_inicio = '2022-01-01'
data_fim = '2022-12-31'

# Defina os símbolos das ações do portfólio
acoes = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA']  # Exemplo de símbolos de ações

# Obtenha os dados históricos das ações do portfólio
dados = yf.download(acoes, start=data_inicio, end=data_fim)

# Calcular os retornos percentuais diários
retornos = dados['Close'].pct_change()

# Calcular a volatilidade (desvio padrão) dos retornos
volatilidades = retornos.std()

# Calcular a matriz de correlação dos retornos
matriz_correlacao = retornos.corr()

# Calcular a covariância dos retornos
covariancia = retornos.cov()

# Supondo que você já tenha os pesos das ações em um vetor
pesos = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Exemplo de pesos para 5 ações

risco_sistematico = np.dot(pesos.T, np.dot(covariancia, pesos))
desvio_padrao_combinado = np.sqrt(risco_sistematico)

retorno_esperado = 0.1  # Exemplo de retorno esperado
retorno_livre_risco = 0.05  # Exemplo de retorno livre de risco

sharpe_ratio = (retorno_esperado - retorno_livre_risco) / desvio_padrao_combinado

nivel_confianca = 0.95
horizonte_tempo = 1  # Em dias

var = desvio_padrao_combinado * np.percentile(retornos.sum(axis=1), (1 - nivel_confianca) * 100) * np.sqrt(horizonte_tempo)

# Gráfico de preços das ações
dados['Close'].plot(title='Preços das Ações', figsize=(10, 6))
plt.ylabel('Preço')
plt.xlabel('Data')
plt.legend(acoes)
plt.show()

# Gráfico de volatilidade das ações
volatilidades.plot(kind='bar', title='Volatilidade das Ações', figsize=(10, 6))
plt.ylabel('Volatilidade')
plt.xlabel('Ações')
plt.show()

# Gráfico de matriz de correlação
plt.imshow(matriz_correlacao, cmap='coolwarm', interpolation='nearest')
plt.title('Matriz de Correlação')
plt.colorbar()
tick_marks = np.arange(len(acoes))
plt.xticks(tick_marks, acoes, rotation=45)
plt.yticks(tick_marks, acoes)
plt.show()


# Calcular a volatilidade (desvio padrão) dos retornos
volatilidades = retornos.std()

# Plotar o gráfico de retorno x risco para cada ação
plt.figure(figsize=(10, 6))
for i in range(len(acoes)):
    retorno = np.mean(retornos[acoes[i]])  # Retorno médio da ação
    risco = volatilidades[acoes[i]]  # Volatilidade da ação
    plt.scatter(risco, retorno, label=acoes[i])

plt.xlabel('Risco (Volatilidade)')
plt.ylabel('Retorno')
plt.title('Retorno x Risco das Ações')
plt.legend()
plt.grid(True)
plt.show()
