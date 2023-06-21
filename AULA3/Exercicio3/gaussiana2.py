import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def diffusion_model(n_samples, n_steps, batch_size):
    # Cria as variáveis para as amostras 
    samples = tf.random.normal(shape=[batch_size, n_samples])
    
    # Define a taxa de difusão (quão rápido o ruído é propagado)
    diffusion_rate = 0.001
    
    # Define a função de difusão (emulação)
    def diffusion_step(x, t):
        noise_t = tf.math.sqrt(diffusion_rate * t) * tf.random.normal(shape=tf.shape(x))
        x_t = tf.math.sqrt(1 - diffusion_rate * t) * x + noise_t
        return x_t
    
    # Executa as etapas de difusão
    for i in range(n_steps):
        print(f"Step {i}/{n_steps}")
        t = (i + 1) / n_steps
        samples = diffusion_step(samples, t)
    
    return samples

# Parâmetros do modelo de difusão
n_samples = 100  # Número de dimensões das amostras
n_steps = 1000  # Número de etapas de difusão
batch_size = 1000  # Tamanho do lote de amostras

# Cria o modelo de difusão
samples = diffusion_model(n_samples, n_steps, batch_size)

# Inicializa as variáveis no TensorFlow
tf.random.set_seed(42)

# Executa o modelo no TensorFlow
generated_samples = samples.numpy()

# Exemplo de uso das amostras geradas
# print("Média das amostras geradas:", np.mean(generated_samples))
# print("Desvio padrão das amostras geradas:", np.std(generated_samples))

# # Plotagem do histograma das amostras geradas
# plt.figure(figsize=(8, 6))
# sns.histplot(generated_samples.flatten(), kde=True, color='skyblue')
# plt.title('Histograma das Amostras Geradas')
# plt.xlabel('Valor')
# plt.ylabel('Contagem')
# plt.show()

# Comparação com a distribuição gaussiana teórica
plt.figure(figsize=(8, 6))
sns.kdeplot(generated_samples.flatten(), color='skyblue', label='Amostras Geradas')
x = np.linspace(np.min(generated_samples), np.max(generated_samples), 100)
gaussian_pdf = np.exp(-0.5 * ((x - np.mean(generated_samples)) / np.std(generated_samples)) ** 2) / (np.sqrt(2 * np.pi) * np.std(generated_samples))
plt.plot(x, gaussian_pdf, color='red', label='Distribuição Gaussiana')
plt.title('Densidade vs. Distribuição Gaussiana')
plt.xlabel('Valor')
plt.ylabel('Densidade')
plt.legend()
plt.show()
