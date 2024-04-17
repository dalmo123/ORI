import sys
import numpy as np
import matplotlib.pyplot as plt


def ler_entrada(arquivo):
    with open(arquivo, 'r') as file:
        n = int(file.readline().strip())
        respostas_ideais = []
        respostas_sistema = []
        for _ in range(n):
            respostas_ideais.append(file.readline().strip().split())
        for _ in range(n):
            respostas_sistema.append(file.readline().strip().split())
    return respostas_ideais, respostas_sistema

def calcular_precisao_revocacao(ideal, recuperado):
    precisoes = []
    revocacoes = []
    relevantes_recuperados = 0
    total_ideal = len(ideal)
    
    for indice, doc in enumerate(recuperado):
        if doc in ideal:
            relevantes_recuperados += 1
        precisao = relevantes_recuperados / (indice + 1)
        revocacao = relevantes_recuperados / total_ideal
        
        # Verifica se houve uma mudança na relevância
        if len(revocacoes) == 0 or revocacao != revocacoes[-1]:
            precisoes.append(precisao)
            revocacoes.append(revocacao)

    return np.array(revocacoes), np.array(precisoes)

def interpolar_precisoes(revocacoes, precisoes):
    niveis_revocacao = np.linspace(0, 1, 11)
    precisoes_interpoladas = [0] * len(niveis_revocacao)
    for i, nivel in enumerate(niveis_revocacao):
        indices = np.where(revocacoes >= nivel)[0]
        if indices.size > 0:
            precisoes_interpoladas[i] = np.max(precisoes[indices[0]:])
    return np.array(niveis_revocacao), np.array(precisoes_interpoladas)

def salvar_media_precisoes(niveis_revocacao, media_precisoes):
    with open('media.txt', 'w') as file:
        for r, p in zip(niveis_revocacao, media_precisoes):
            file.write(f"{p:.2f}\n")

def plot_precisao_revocacao(niveis_revocacao, precisoes_interpoladas, titulo):
    plt.figure(figsize=(10, 5))
    plt.plot(niveis_revocacao, precisoes_interpoladas, marker='o', linestyle='-')
    plt.xlabel('Revocação')
    plt.ylabel('Precisão')
    plt.title(titulo)
    plt.grid(True)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 avaliacao.py <arquivo_entrada>")
        sys.exit(1)
    arquivo_entrada = sys.argv[1]
    respostas_ideais, respostas_sistema = ler_entrada(arquivo_entrada)
    consultas_precisao_revocacao = []

    for i, (ideal, sistema) in enumerate(zip(respostas_ideais, respostas_sistema)):
        revocacoes, precisoes = calcular_precisao_revocacao(ideal, sistema)
        niveis_revocacao, precisoes_interpoladas = interpolar_precisoes(revocacoes, precisoes)
        consultas_precisao_revocacao.append(precisoes_interpoladas)
        
        plot_precisao_revocacao(niveis_revocacao, precisoes_interpoladas, f'Precisão x Revocação para consulta {i}')

    # Calcula a média das precisões para cada nível de revocação
    media_precisoes = np.mean(consultas_precisao_revocacao, axis=0)
    
    salvar_media_precisoes(niveis_revocacao, media_precisoes)

    plot_precisao_revocacao(niveis_revocacao, media_precisoes, 'Precisão média para todas as consultas')

if __name__ == "__main__":
    main()