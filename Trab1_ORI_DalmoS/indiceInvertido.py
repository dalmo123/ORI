import spacy
import sys
import os

# Carregando o modelo de língua portuguesa do SpaCy
nlp = spacy.load('pt_core_news_lg')


def processar_frase(frase):
    # Converte a frase para minúsculas e usa SpaCy para tokenização
    tokens = nlp(frase.lower())

    # Lista para armazenar as palavras lematizadas
    palavras_lematizadas = []

    # Itera sobre os tokens
    for token in tokens:
        # Converte o token em string
        token_str = str(token.lemma_.lower())

        # Verifica se o token não é uma pontuação, não é uma stopword e não é uma palavra composta
        if not token.is_punct and not token.is_stop and len(token_str.split()) == 1:
            # Lematiza o token e adiciona o lemma à lista de palavras lematizadas
            palavras_lematizadas.append(token_str)

    return palavras_lematizadas



def criar_indice_invertido(documentos):
    indice_invertido = {}
    
    for doc_id, documento in enumerate(documentos, start=1):
        palavras = processar_frase(documento)
        for palavra in palavras:
            if palavra not in indice_invertido:
                indice_invertido[palavra] = {doc_id: 1}
            else:
                if doc_id not in indice_invertido[palavra]:
                    indice_invertido[palavra][doc_id] = 1
                else:
                    indice_invertido[palavra][doc_id] += 1
    
    # Ordenar as palavras do índice em ordem alfabética
    indice_invertido_ordenado = {palavra: indice_invertido[palavra] for palavra in sorted(indice_invertido)}
    
    return indice_invertido_ordenado

def ler_arquivo(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()
    return [linha.strip() for linha in linhas]

def salvar_indice(indice, caminho_saida='indice.txt'):
    with open(caminho_saida, 'w', encoding='utf-8') as arquivo_saida:
        for palavra, ocorrencias in indice.items():
            arquivo_saida.write(f'{palavra}: {ocorrencias}\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso correto: python indiceInvertido.py caminho_do_arquivo.txt")
        sys.exit(1)
    
    caminho_arquivo = sys.argv[1]
    # Obtém o caminho absoluto do diretório atual
    diretorio_atual = os.getcwd()
    
    # Define o caminho absoluto do arquivo "qualquer base"
    caminho_arquivo_base = os.path.join(diretorio_atual, caminho_arquivo)

    # Verifica se o arquivo existe
    if not os.path.exists(caminho_arquivo_base):
        print(f"O arquivo {caminho_arquivo_base} não foi encontrado.")
        sys.exit(1)

    caminhos_documentos = ler_arquivo(caminho_arquivo_base)

    documentos = []
    for caminho_documento in caminhos_documentos:
        caminho_completo_documento = os.path.join(diretorio_atual, caminho_documento)  # Adiciona o caminho absoluto do documento
        with open(caminho_completo_documento, 'r', encoding='utf-8') as arquivo_documento:
            documento = arquivo_documento.read()
            documentos.append(documento)

    indice = criar_indice_invertido(documentos)

    # Exibindo o índice invertido
    for palavra, ocorrencias in indice.items():
        print(f'{palavra}: {ocorrencias}')

    # Salvando o índice no arquivo indice.txt
    salvar_indice(indice)
