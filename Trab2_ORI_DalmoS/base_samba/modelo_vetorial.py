import math
import spacy
import sys
import os
from collections import defaultdict

# Carrega o modelo de língua portuguesa do SpaCy
nlp = spacy.load('pt_core_news_lg')


def processar_frase(frase):
    tokens = nlp(frase.lower())

    contagem_palavras = defaultdict(int)  # Dicionário para armazenar as contagens de cada palavra lematizada

    for token in tokens:
        token_str = str(token.lemma_.lower())

        if not token.is_punct and not token.is_stop and len(token_str.split()) == 1:
            contagem_palavras[token_str] += 1

    return contagem_palavras



def criar_indice_invertido(documentos):
    indice_invertido = defaultdict(dict)
    
    for doc_id, documento in enumerate(documentos, start=1):
        palavras = processar_frase(documento)  # Usar palavras processadas
        for palavra, contagem in palavras.items():  # Iterar sobre as palavras e suas contagens
            if doc_id not in indice_invertido[palavra]:
                indice_invertido[palavra][doc_id] = contagem  # Adicionar a contagem ao índice invertido
            else:
                indice_invertido[palavra][doc_id] += contagem  # Incrementar a contagem no índice invertido
    
    return indice_invertido



def calcular_IDF(indice_invertido, total_documentos):
    idf = {}
    for termo, ocorrencias in indice_invertido.items():
        idf[termo] = math.log10(total_documentos / len(ocorrencias))
    return idf

def calcular_pesos_TF_IDF(indice_invertido, idf):
    pesos = defaultdict(dict)
    for termo, ocorrencias in indice_invertido.items():
        for doc_id, freq in ocorrencias.items():
            TF = 1 + math.log10(freq)
            pesos[doc_id][termo] = TF * idf[termo]
    return pesos


def calcular_tf_idf_consulta(consulta_processada, idf, total_termos_consulta):
    tf_idf_consulta = {}
    for termo, freq in consulta_processada.items():
        tf = freq / total_termos_consulta  # Calcula o TF em relação ao total de termos na consulta
        if termo in idf:
            tf_idf_consulta[termo] = tf * idf[termo] 
    return tf_idf_consulta



def calcular_similaridade(consulta_processada, pesos, tf_idf_consulta):
    similaridade = defaultdict(float)
    for doc_id, peso_termos in pesos.items():
        dot_product = sum(peso_termos[p] * tf_idf_consulta[p] for p in peso_termos if p in tf_idf_consulta)
        magnitude_pesos = math.sqrt(sum(v ** 2 for v in peso_termos.values()))
        magnitude_consulta = math.sqrt(sum(v ** 2 for v in tf_idf_consulta.values()))
        similaridade[doc_id] = dot_product / (magnitude_pesos * magnitude_consulta) if magnitude_pesos * magnitude_consulta != 0 else 0
    return similaridade

def salvar_indice(indice, caminho_saida='indice.txt'):
    with open(caminho_saida, 'w', encoding='utf-8') as arquivo_saida:
        for palavra, ocorrencias in sorted(indice.items()):
            arquivo_saida.write(f'{palavra}: {ocorrencias}\n')
            

def salvar_pesos(pesos, caminhos_documentos, caminho_saida='pesos.txt'):
    with open(caminho_saida, 'w', encoding='utf-8') as arquivo_saida:
        for doc_id, peso_termos in pesos.items():
            doc_nome = caminhos_documentos[doc_id - 1]  # Obtém o nome do documento a partir do índice
            arquivo_saida.write(f'{doc_nome}: ')
            termos_pesos = [f'{termo},{peso}' for termo, peso in sorted(peso_termos.items())]
            arquivo_saida.write(' '.join(termos_pesos))
            arquivo_saida.write('\n')



def salvar_resposta(similaridade, limite=0.001, caminho_saida='resposta.txt', caminhos_documentos=None):
    relevantes = [(caminhos_documentos[doc_id - 1], sim) for doc_id, sim in similaridade.items() if sim >= limite]
    relevantes_ordenados = sorted(relevantes, key=lambda x: x[1], reverse=True)
    with open(caminho_saida, 'w', encoding='utf-8') as arquivo_saida:
        arquivo_saida.write(f'{len(relevantes_ordenados)}\n')
        for doc_nome, sim in relevantes_ordenados:
            arquivo_saida.write(f'{doc_nome} {sim}\n')


def ler_arquivo(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()
    return [linha.strip() for linha in linhas]



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso correto: python modelo_vetorial.py caminho_do_arquivo_base caminho_do_arquivo_consulta")
        sys.exit(1)
    
    caminho_arquivo_base = sys.argv[1]
    caminho_arquivo_consulta = sys.argv[2]

    # Obtém o caminho absoluto do diretório atual
    diretorio_atual = os.getcwd()
    
    # Define o caminho absoluto do arquivo da base
    caminho_arquivo_base = os.path.join(diretorio_atual, caminho_arquivo_base)

    # Define o caminho absoluto do arquivo de consulta
    caminho_arquivo_consulta = os.path.join(diretorio_atual, caminho_arquivo_consulta)

    # Verifica se os arquivos existem
    if not os.path.exists(caminho_arquivo_base) or not os.path.exists(caminho_arquivo_consulta):
        print("Um ou ambos os arquivos não foram encontrados.")
        sys.exit(1)

    # Lê os documentos da base
    caminhos_documentos = ler_arquivo(caminho_arquivo_base)
    documentos = []
    for caminho_documento in caminhos_documentos:
        caminho_completo_documento = os.path.join(diretorio_atual, caminho_documento)
        with open(caminho_completo_documento, 'r', encoding='utf-8') as arquivo_documento:
            documento = arquivo_documento.read()
            documentos.append(documento)

     # Cria o índice invertido com base nos documentos da base
    indice_invertido = criar_indice_invertido(documentos)

    # Calcula o número total de documentos na base
    total_documentos = len(documentos)

    # Calcula o IDF usando o índice invertido
    idf = calcular_IDF(indice_invertido, total_documentos)

    # Calcula os pesos TF-IDF usando o índice invertido e o IDF
    pesos = calcular_pesos_TF_IDF(indice_invertido, idf)

    # Lê a consulta do arquivo
    consulta = ler_arquivo(caminho_arquivo_consulta)
    consulta_processada = processar_frase(consulta[0])  # Assumindo que há apenas uma consulta no arquivo

    # Calcula a quantidade total de termos na consulta
    total_termos_consulta = sum(consulta_processada.values())

    # Calcula o TF-IDF da consulta
    tf_idf_consulta = calcular_tf_idf_consulta(consulta_processada, idf, total_termos_consulta)

    # Calcula a similaridade entre a consulta e os documentos usando os pesos TF-IDF e a fórmula do cosseno
    similaridade = calcular_similaridade(consulta_processada, pesos, tf_idf_consulta)

    # Salva a resposta no arquivo
    salvar_resposta(similaridade, caminhos_documentos=caminhos_documentos)

    # Salva o índice invertido no arquivo
    salvar_indice(indice_invertido)

    # Salva os pesos TF-IDF no arquivo
    salvar_pesos(pesos, caminhos_documentos)