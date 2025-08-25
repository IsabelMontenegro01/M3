# Importação das bibliotecas necessárias
import pandas as pd  # Para manipulação de dados
from sklearn.model_selection import train_test_split  # Para separar dados em treino e teste
from sklearn.neighbors import KNeighborsRegressor  # Para o modelo KNN de regressão
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Métricas de regressão
import matplotlib.pyplot as plt  # Para plots opcionais de análise

# Passo 1: Carregar o dataset
# O arquivo CSV deve estar na mesma pasta que este script
df = pd.read_csv('dataset_educacao_graduacao_brasil_500.csv')

# Passo 2: Pré-processamento dos dados
# Limpar e converter colunas numéricas (algumas podem ter formatação errada, como '04.09' em vez de 4.09)
df['Renda_Familiar_SM'] = df['Renda_Familiar_SM'].astype(str).str.replace(',', '.').astype(float)  # Corrige vírgulas para pontos
df['CRA'] = df['CRA'].astype(str).str.replace(',', '.').astype(float)  # Corrige CRA se necessário (ex: '05.06' -> 5.06)
df['Nota_ENEM'] = df['Nota_ENEM'].astype(float)  # Garante que seja float
df['Horas_Estudo_Semanais'] = df['Horas_Estudo_Semanais'].astype(float)

# Criar variável binária para bolsistas: 1 se tem bolsa (Prouni, FIES, Bolsa Institucional), 0 se Nenhum
df['Tem_Bolsa'] = df['Bolsa_ou_Financiamento'].apply(lambda x: 1 if x in ['Prouni', 'FIES', 'Bolsa Institucional'] else 0)

# Selecionar features relevantes para a regressão: Renda_Familiar_SM, Nota_ENEM, Horas_Estudo_Semanais
# Target: CRA (Coeficiente de Rendimento Acadêmico)
features = ['Renda_Familiar_SM', 'Nota_ENEM', 'Horas_Estudo_Semanais']
X = df[features]
y = df['CRA']

# Passo 3: Separar dados em treino e teste (80% treino, 20% teste) para evitar overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # random_state para reproducibilidade

# Passo 4: Treinar e avaliar o modelo KNN para regressão
# Vamos variar K (número de vizinhos) para discutir overfitting (K pequeno) vs underfitting (K grande)
ks = [3, 5, 7, 10]  # Valores de K a testar
results = {}  # Dicionário para armazenar métricas

for k in ks:
    # Criar e treinar o modelo KNN com métrica de distância euclidiana (padrão)
    knn = KNeighborsRegressor(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    
    # Prever no conjunto de teste
    y_pred = knn.predict(X_test)
    
    # Calcular métricas de regressão
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Armazenar resultados
    results[k] = {'MAE': mae, 'MSE': mse, 'R2': r2}
    
    # Imprimir métricas para cada K
    print(f"\nResultados para K={k}:")
    print(f"MAE: {mae:.2f} (Erro médio absoluto)")
    print(f"MSE: {mse:.2f} (Erro quadrático médio)")
    print(f"R²: {r2:.2f} (Coeficiente de determinação - quanto maior, melhor o ajuste)")

# Discussão: Overfitting/Underfitting
# - K pequeno (ex: 3): Pode overfit, capturando ruído (baixa generalização, mas bom em treino).
# - K grande (ex: 10): Pode underfit, sendo muito genérico (alta bias).
# Escolha o K com melhor R² no teste. Aqui, testamos empiricamente.

# Passo 5: Análise por subconjuntos para testar a hipótese
# 5.1: Comparar CRA médio entre bolsistas e não bolsistas
bolsistas = df[df['Tem_Bolsa'] == 1]
nao_bolsistas = df[df['Tem_Bolsa'] == 0]

print("\nAnálise por subconjuntos (Hipótese: Bolsistas têm CRA semelhante ou superior):")
print(f"CRA médio bolsistas: {bolsistas['CRA'].mean():.2f}")
print(f"CRA médio não bolsistas: {nao_bolsistas['CRA'].mean():.2f}")

# Treinar KNN separado para bolsistas e não bolsistas (para importância empírica)
# Para bolsistas
X_bols = bolsistas[features]
y_bols = bolsistas['CRA']
if len(X_bols) > 0:  # Verificar se há dados
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_bols, y_bols, test_size=0.2, random_state=42)
    knn_b = KNeighborsRegressor(n_neighbors=5)  # Usando K=5 como exemplo
    knn_b.fit(X_b_train, y_b_train)
    y_b_pred = knn_b.predict(X_b_test)
    print(f"\nMétricas KNN para bolsistas (R²: {r2_score(y_b_test, y_b_pred):.2f})")

# Para não bolsistas (similar)
X_nao = nao_bolsistas[features]
y_nao = nao_bolsistas['CRA']
if len(X_nao) > 0:
    X_n_train, X_n_test, y_n_train, y_n_test = train_test_split(X_nao, y_nao, test_size=0.2, random_state=42)
    knn_n = KNeighborsRegressor(n_neighbors=5)
    knn_n.fit(X_n_train, y_n_train)
    y_n_pred = knn_n.predict(X_n_test)
    print(f"Métricas KNN para não bolsistas (R²: {r2_score(y_n_test, y_n_pred):.2f})")

# 5.2: Análise por Tipo_IES (Pública vs. Privada)
publicas = df[df['Tipo_IES'] == 'Pública']
privadas = df[df['Tipo_IES'] == 'Privada']

print("\nAnálise por Tipo_IES:")
print(f"CRA médio em Públicas: {publicas['CRA'].mean():.2f} (Bolsistas: {publicas[publicas['Tem_Bolsa']==1]['CRA'].mean():.2f})")
print(f"CRA médio em Privadas: {privadas['CRA'].mean():.2f} (Bolsistas: {privadas[privadas['Tem_Bolsa']==1]['CRA'].mean():.2f})")

# Treinar KNN por Tipo_IES (similar ao acima, para verificar se o padrão se mantém)
# Para Públicas
X_pub = publicas[features]
y_pub = publicas['CRA']
if len(X_pub) > 0:
    X_pub_train, X_pub_test, y_pub_train, y_pub_test = train_test_split(X_pub, y_pub, test_size=0.2, random_state=42)
    knn_pub = KNeighborsRegressor(n_neighbors=5)
    knn_pub.fit(X_pub_train, y_pub_train)
    y_pub_pred = knn_pub.predict(X_pub_test)
    print(f"Métricas KNN para Públicas (R²: {r2_score(y_pub_test, y_pub_pred):.2f})")

# Para Privadas
X_priv = privadas[features]
y_priv = privadas['CRA']
if len(X_priv) > 0:
    X_priv_train, X_priv_test, y_priv_train, y_priv_test = train_test_split(X_priv, y_priv, test_size=0.2, random_state=42)
    knn_priv = KNeighborsRegressor(n_neighbors=5)
    knn_priv.fit(X_priv_train, y_priv_train)
    y_priv_pred = knn_priv.predict(X_priv_test)
    print(f"Métricas KNN para Privadas (R²: {r2_score(y_priv_test, y_priv_pred):.2f})")

# Passo 6: Interpretação dos resultados à luz da hipótese
# - Se CRA médio de bolsistas >= não bolsistas, e R² similar/bom nos subconjuntos, a hipótese se confirma.
# - Verifique se o padrão (bolsistas com CRA melhor) se mantém em Públicas e Privadas.
# - Se R² baixo, o modelo pode não capturar bem (underfitting); ajuste features ou K.
# Opcional: Plotar dispersão para visualização
plt.scatter(df['Nota_ENEM'], df['CRA'], c=df['Tem_Bolsa'], cmap='viridis')
plt.xlabel('Nota ENEM')
plt.ylabel('CRA')
plt.title('CRA vs Nota ENEM (Colorido por Bolsa)')
plt.show()
