import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    plt = None
    SEABORN_AVAILABLE = False

try:
    from scipy import stats
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    np = None
    SCIPY_AVAILABLE = False

BASE_URL = "https://dadosabertos.compras.gov.br"
TIMEOUT = 30
MAX_RETRIES = 3
PAGE_SIZE = 100
DEFAULT_MAX_PAGES = None
SESSION = requests.Session()

# Filtros padrão iniciais usados como template. O usuário deve fornecer valores reais
# antes de chamar a API, evitando consultas muito amplas e pesadas.
DEFAULT_LICITACAO_FILTERS = {
    "uasg": None,
    "modalidade": None,
    "data_publicacao_inicial": None,
    "data_publicacao_final": None,
    "pertence14133": None,
}

DEFAULT_ITEM_FILTERS = {
    "uasg": None,
    "modalidade": None,
    "codigo_item_material": None,
    "codigo_item_servico": None,
    "cnpj_fornecedor": None,
    "cpfVencedor": None,
}

# Funções utilitárias para normalizar parâmetros e inspecionar dados de resposta.
# Estas funções ajudam a manter os filtros limpos e a extrair campos mesmo quando
# o JSON de resposta usa nomes de chave diferentes entre endpoints.

def normalize_params(params):
    """Normaliza parâmetros antes de enviar para o endpoint.

    - Remove valores None.
    - Ignora strings vazias.
    - Converte booleanos para o formato aceito pela API.
    """
    result = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            result[key] = str(value).lower()
        elif isinstance(value, str) and value.strip() == "":
            continue
        else:
            result[key] = value
    return result


def valor_preenchido(valor):
    """Retorna True quando o valor é efetivamente preenchido."""
    return valor is not None and (not isinstance(valor, str) or valor.strip() != "")


def valor_campo(registro, *campos):
    """Retorna o primeiro campo presente e não vazio de um registro."""
    for campo in campos:
        valor = registro.get(campo)
        if valor_preenchido(valor):
            return valor
    return None


def filtrar_registros_completos(registros, tipo):
    """Filtra registros para manter apenas os que têm campos essenciais preenchidos."""
    filtrados = []
    for registro in registros:
        if tipo == "licitacao":
            situacao = valor_campo(registro, "situacao", "situacaoLicitacao")
            objeto = valor_campo(registro, "objeto", "objetoLicitacao", "descricao_item", "descricaoItem")
            numero = valor_campo(registro, "numero_aviso", "numeroAviso")
            uasg = valor_campo(registro, "uasg", "uasgCodigo")
            if valor_preenchido(situacao) and valor_preenchido(objeto) and valor_preenchido(numero) and valor_preenchido(uasg):
                filtrados.append(registro)
        elif tipo == "item":
            descricao = valor_campo(registro, "descricao_item", "descricaoItem")
            quantidade = valor_campo(registro, "quantidade", "qtde")
            valor_estimado = valor_campo(registro, "valor_estimado", "valorEstimado")
            fornecedor = valor_campo(registro, "cnpj_fornecedor", "cnpjFornecedor")
            if valor_preenchido(descricao) and valor_preenchido(quantidade) and valor_preenchido(valor_estimado) and valor_preenchido(fornecedor):
                filtrados.append(registro)
    return filtrados


def clean_item_params(params):
    """Remove parâmetros não suportados pelo endpoint de itens."""
    invalid = ["decreto_7174"]
    return {k: v for k, v in params.items() if k not in invalid and v is not None}


def extrair_html_resumo(html_text):
    """Usa BeautifulSoup para extrair um resumo útil de uma resposta HTML."""
    if not html_text:
        return "Sem conteúdo HTML retornado."
    if not BS4_AVAILABLE:
        return html_text[:300]

    soup = BeautifulSoup(html_text, "html.parser")
    titulo = soup.title.get_text(" ", strip=True) if soup.title else "Sem título"
    textos = [tag.get_text(" ", strip=True) for tag in soup.find_all(["p", "h1", "h2", "li"])][:5]
    textos = [texto for texto in textos if texto]
    resumo = " | ".join(textos[:5])
    return f"Título: {titulo}. Fragmentos: {resumo[:400]}"


def requisicao_json(path, params=None):
    """Faz uma requisição GET e trata erros comuns de HTTP."""
    url = f"{BASE_URL}{path}"
    params = normalize_params(params)

    for tentativa in range(MAX_RETRIES):
        try:
            resposta = SESSION.get(url, params=params, timeout=TIMEOUT)
            if resposta.status_code in {400, 404}:
                raise RuntimeError(f"{resposta.status_code} {resposta.reason}: {resposta.text}")
            resposta.raise_for_status()

            try:
                return resposta.json()
            except ValueError:
                html_summary = extrair_html_resumo(resposta.text)
                raise RuntimeError(
                    f"Resposta da API não retornou JSON válido para {url}. "
                    f"Possível conteúdo HTML ou pagina de erro. Detalhes: {html_summary}"
                ) from None
        except RuntimeError:
            raise
        except requests.RequestException as erro:
            if tentativa == MAX_RETRIES - 1:
                raise RuntimeError(f"Erro na requisição para {url}: {erro}") from erro
            time.sleep(1 + tentativa)


def paginar_endpoint(path, params=None, max_pages=DEFAULT_MAX_PAGES):
    """Faz paginação do endpoint e coleta metadados de execução.

    Quando max_pages for None, o script continua até o servidor informar que não há
    mais páginas, seguindo uma estratégia mais "web scraping" e menos agressiva.
    """
    params = dict(params or {})
    pagina = 1
    todos = []
    metadata = {
        "totalRegistros": 0,
        "totalPaginas": 0,
        "paginasRestantes": 0,
        "fetchedPages": 0,
        "limitReached": False,
    }

    while True:
        atual = dict(params)
        atual.update({"pagina": pagina, "tamanhoPagina": PAGE_SIZE})
        dados = requisicao_json(path, atual)

        resultado = dados.get("resultado", [])
        todos.extend(resultado)
        metadata["totalRegistros"] = dados.get("totalRegistros", metadata["totalRegistros"])
        metadata["totalPaginas"] = dados.get("totalPaginas", metadata["totalPaginas"])
        metadata["paginasRestantes"] = dados.get("paginasRestantes", metadata["paginasRestantes"])
        metadata["fetchedPages"] = pagina

        if metadata["totalPaginas"] and pagina >= metadata["totalPaginas"]:
            break
        if metadata["paginasRestantes"] <= 0:
            break
        if max_pages is not None and pagina >= max_pages:
            metadata["limitReached"] = True
            break

        pagina += 1

    return todos, metadata


def consultar_licitacoes(filtros=None, modo="completos", max_pages=DEFAULT_MAX_PAGES):
    """Consulta licitações e aplica fallback em casos de rejeição de modalidade."""
    params = dict(DEFAULT_LICITACAO_FILTERS)
    if filtros:
        params.update(filtros)

    try:
        resultados, metadata = paginar_endpoint("/modulo-legado/1_consultarLicitacao", params, max_pages)
    except RuntimeError as erro:
        if "400" in str(erro) and params.get("modalidade") is not None:
            print("Consulta principal rejeitou a modalidade. Tentando endpoint de itens de licitação como fallback...")
            itens, metadata = consultar_itens_licitacao(params, "todos", max_pages)
            resultados = [
                {
                    "numero_aviso": item.get("numero_aviso") or item.get("numeroAviso"),
                    "modalidade": item.get("modalidade"),
                    "nome_modalidade": item.get("nome_modalidade"),
                    "uasg": item.get("uasg"),
                    "objeto": item.get("descricao_item") or item.get("nome_material") or item.get("nome_servico") or item.get("descricaoItem"),
                    "data_publicacao": item.get("data_publicacao") or item.get("dataPublicacao") or "",
                    "situacao": "",
                    "descricao_item": item.get("descricao_item") or item.get("descricaoItem"),
                    "quantidade": item.get("quantidade") or item.get("qtde"),
                    "valor_estimado": item.get("valor_estimado") or item.get("valorEstimado"),
                    "cnpj_fornecedor": item.get("cnpj_fornecedor") or item.get("cnpjFornecedor"),
                    "_origem": "item_licitacao",
                }
                for item in itens
            ]
        else:
            raise

    if modo == "completos":
        resultados = filtrar_registros_completos(resultados, "licitacao")
    return resultados, metadata


def consultar_itens_licitacao(filtros=None, modo="completos", max_pages=DEFAULT_MAX_PAGES):
    """Consulta o endpoint de itens, aplica limpeza de parâmetros e opcionalmente filtra completos."""
    params = dict(DEFAULT_ITEM_FILTERS)
    if filtros:
        params.update(filtros)
    # Remove parâmetros conhecidos que causam falha no endpoint de itens.
    params = clean_item_params(params)

    resultados, metadata = paginar_endpoint("/modulo-legado/2_consultarItemLicitacao", params, max_pages)
    if modo == "completos":
        resultados = filtrar_registros_completos(resultados, "item")
    return resultados, metadata


def registros_para_dataframe(registros):
    """Converte lista de registros em DataFrame pandas."""
    if not PANDAS_AVAILABLE:
        raise RuntimeError("Pandas não está instalado. Instale com 'pip install pandas' para usar DataFrame.")
    if not registros:
        return pd.DataFrame()
    df = pd.json_normalize(registros)
    return df


def salvar_csv(arquivo, registros, campos):
    if PANDAS_AVAILABLE:
        df = registros_para_dataframe(registros)
        if campos:
            cols = [c for c in campos if c in df.columns]
            df.to_csv(arquivo, columns=cols, index=False, encoding="utf-8")
        else:
            df.to_csv(arquivo, index=False, encoding="utf-8")
    else:
        with open(arquivo, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=campos)
            writer.writeheader()
            for registro in registros:
                writer.writerow({campo: registro.get(campo, "") for campo in campos})
    print(f"Arquivo salvo em: {arquivo}")


def salvar_json(arquivo, registros):
    if PANDAS_AVAILABLE:
        df = registros_para_dataframe(registros)
        df.to_json(arquivo, orient="records", force_ascii=False, indent=2)
    else:
        with open(arquivo, "w", encoding="utf-8") as handle:
            json.dump(registros, handle, ensure_ascii=False, indent=2)
    print(f"Arquivo salvo em: {arquivo}")

def detectar_anomalias_scipy(df):
    """Usa SciPy para encontrar outliers nos valores estimados."""
    if not SCIPY_AVAILABLE or df.empty or 'valor_estimado' not in df.columns:
        return df

    df_clean = df.copy()
    df_clean['valor_estimado_num'] = pd.to_numeric(df_clean['valor_estimado'], errors='coerce')
    df_clean = df_clean.dropna(subset=['valor_estimado_num'])

    if df_clean.empty:
        return df_clean

    z_scores = np.abs(stats.zscore(df_clean['valor_estimado_num']))
    outliers = df_clean[z_scores > 3]
    
    if not outliers.empty:
        print(f"\n[ALERTA] SciPy detectou {len(outliers)} itens com valores atípicos (outliers):")
        for _, row in outliers.head(5).iterrows():
            print(f" - Item {row.get('codigo_item_material', 'N/A')}: R$ {row['valor_estimado_num']:.2f}")
            
    return outliers

def visualizar_resultados_seaborn(df):
    """Gera visualizações analíticas com Seaborn."""
    if not SEABORN_AVAILABLE or df.empty:
        print("Seaborn não está disponível ou DataFrame está vazio.")
        return

    if 'valor_estimado' in df.columns:
        df['valor_estimado_num'] = pd.to_numeric(df['valor_estimado'], errors='coerce')

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    if 'valor_estimado_num' in df.columns and not df['valor_estimado_num'].dropna().empty:
        sns.histplot(data=df, x='valor_estimado_num', kde=True, bins=20, color='blue', ax=ax[0])
        ax[0].set_title('Distribuição de Valores Estimados', fontsize=14)
        ax[0].set_xlabel('Valor (R$)')
        ax[0].set_ylabel('Frequência')
        ax[0].set_xscale('log') 

    if 'cnpj_fornecedor' in df.columns:
        top_fornecedores = df['cnpj_fornecedor'].value_counts().head(5).reset_index()
        top_fornecedores.columns = ['cnpj_fornecedor', 'contagem']
        
        sns.barplot(data=top_fornecedores, y='cnpj_fornecedor', x='contagem', palette='viridis', ax=ax[1])
        ax[1].set_title('Top 5 Fornecedores por Volume de Itens', fontsize=14)
        ax[1].set_xlabel('Quantidade de Itens')
        ax[1].set_ylabel('CNPJ do Fornecedor')

    plt.tight_layout()
    plt.show()

def resumo_metadata(metadata):
    lines = [
        f"Total registros API: {metadata.get('totalRegistros', 0)}",
        f"Total páginas API: {metadata.get('totalPaginas', 0)}",
        f"Páginas consultadas: {metadata.get('fetchedPages', 0)}",
        f"Registros retornados: {metadata.get('fetchedPages', 0) * PAGE_SIZE if metadata.get('fetchedPages') else 0}",
    ]
    if metadata.get("limitReached"):
        lines.append("Limite de páginas atingido; a consulta foi interrompida para evitar download enorme.")
    return "\n".join(lines)


def imprimir_licitacoes(licitacoes, metadata=None, modo="completos"):
    if metadata:
        print("\nResumo da consulta:")
        print(resumo_metadata(metadata))

    if not licitacoes:
        print("Nenhuma licitação encontrada.")
        return

    label = "completos" if modo == "completos" else "todos"
    print(f"\nEncontradas {len(licitacoes)} licitações ({label}):\n")
    for idx, lic in enumerate(licitacoes, start=1):
        print(f"   UASG: {lic.get('uasg') or lic.get('uasgCodigo') or '---'}")
        print(f"   UASG nome: {lic.get('nome_uasg') or '---'}")
        print(f"   Objeto: {lic.get('objeto') or lic.get('objetoLicitacao') or '---'}")
        print(f"   Data publicação: {lic.get('data_publicacao') or lic.get('dataPublicacao') or '---'}")
        print(f"   Situação: {lic.get('situacao') or lic.get('situacaoLicitacao') or '---'}")
        if lic.get("_origem") == "item_licitacao":
            print(f"   Item: {lic.get('descricao_item') or '---'}")
            print(f"   Quantidade: {lic.get('quantidade') or '---'}")
            print(f"   Valor estimado: {lic.get('valor_estimado') or '---'}")
            print(f"   Fornecedor: {lic.get('cnpj_fornecedor') or '---'}")
        print()


def imprimir_itens(itens, metadata=None, modo="completos"):
    if metadata:
        print("\nResumo da consulta:")
        print(resumo_metadata(metadata))

    if not itens:
        print("Nenhum item encontrado.")
        return

    label = "completos" if modo == "completos" else "todos"
    print(f"\nEncontrados {len(itens)} itens de licitação ({label}):\n")
    for idx, item in enumerate(itens, start=1):
        print(f"{idx}. Item: {item.get('codigo_item_material') or item.get('codigoItemMaterial') or '---'}")
        print(f"   Descrição: {item.get('descricao_item') or item.get('descricaoItem') or '---'}")
        print(f"   Quantidade: {item.get('quantidade') or item.get('qtde') or '---'}")
        print(f"   Valor estimado: {item.get('valor_estimado') or item.get('valorEstimado') or '---'}")
        print(f"   Fornecedor: {item.get('cnpj_fornecedor') or item.get('cnpjFornecedor') or '---'}")
        print()


def gerar_nome_arquivo(tipo, formato, filtros):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    partes = [tipo]
    if filtros.get("uasg"):
        partes.append(f"uasg{filtros['uasg']}")
    if filtros.get("modalidade"):
        partes.append(f"mod{filtros['modalidade']}")
    partes.append(stamp)
    return Path(".").joinpath("_".join(partes) + f".{formato}")


def pedir_input(nome, tipo=str, padrao=None, obrigatorio=False):
    while True:
        prompt = f"{nome} [{padrao}]: " if padrao is not None else f"{nome}: "
        valor = input(prompt).strip()
        if valor == "":
            if obrigatorio:
                print(f"O campo {nome} é obrigatório.")
                continue
            return padrao
        try:
            return tipo(valor)
        except ValueError:
            print(f"Valor inválido para {nome}. Tente novamente.")


def solicitar_filtros_licitacao():
    filtros = {}
    print("Informe os filtros obrigatórios para consulta de licitações.")
    print("É necessário informar pelo menos uma das opções: uasg ou modalidade.")
    print("As datas de publicação também são obrigatórias.")

    filtros["uasg"] = pedir_input("uasg", int, None, obrigatorio=False)
    filtros["modalidade"] = pedir_input("modalidade", int, None, obrigatorio=False)
    filtros["data_publicacao_inicial"] = pedir_input("data_publicacao_inicial", str, None, obrigatorio=True)
    filtros["data_publicacao_final"] = pedir_input("data_publicacao_final", str, None, obrigatorio=True)

    while not any(filtros.get(key) is not None for key in ["uasg", "modalidade"]):
        print("Você deve informar pelo menos uma das opções: uasg, modalidade.")
        filtros["uasg"] = pedir_input("uasg", int, None, obrigatorio=False)
        filtros["modalidade"] = pedir_input("modalidade", int, None, obrigatorio=False)

    valor = input("pertence14133 [false]: ").strip().lower()
    if valor in {"true", "1", "sim", "s", "yes", "y"}:
        filtros["pertence14133"] = True
    elif valor in {"false", "0", "nao", "n", "no"}:
        filtros["pertence14133"] = False

    return filtros


def solicitar_filtros_item():
    filtros = {}
    print("Informe pelo menos um filtro obrigatório para consulta de itens de licitação.")
    print("Use pelo menos um de: uasg, modalidade, codigo_item_material, codigo_item_servico, cnpj_fornecedor ou cpfVencedor.")

    filtros["uasg"] = pedir_input("uasg", int, None, obrigatorio=False)
    filtros["modalidade"] = pedir_input("modalidade", int, None, obrigatorio=False)
    filtros["codigo_item_material"] = pedir_input("codigo_item_material", int, None, obrigatorio=False)
    filtros["codigo_item_servico"] = pedir_input("codigo_item_servico", int, None, obrigatorio=False)
    filtros["cnpj_fornecedor"] = pedir_input("cnpj_fornecedor", str, None, obrigatorio=False)
    filtros["cpfVencedor"] = pedir_input("cpfVencedor", str, None, obrigatorio=False)

    while not any(
        filtros.get(key) is not None
        for key in [
            "uasg",
            "modalidade",
            "codigo_item_material",
            "codigo_item_servico",
            "cnpj_fornecedor",
            "cpfVencedor",
        ]
    ):
        print("Informe pelo menos um filtro obrigatório para itens de licitação.")
        filtros["uasg"] = pedir_input("uasg", int, None, obrigatorio=False)
        filtros["modalidade"] = pedir_input("modalidade", int, None, obrigatorio=False)
        filtros["codigo_item_material"] = pedir_input("codigo_item_material", int, None, obrigatorio=False)
        filtros["codigo_item_servico"] = pedir_input("codigo_item_servico", int, None, obrigatorio=False)
        filtros["cnpj_fornecedor"] = pedir_input("cnpj_fornecedor", str, None, obrigatorio=False)
        filtros["cpfVencedor"] = pedir_input("cpfVencedor", str, None, obrigatorio=False)

    return filtros


def criar_parser():
    """Define argumentos de linha de comando para uso não interativo."""
    parser = argparse.ArgumentParser(description="Consulta licitações e itens de licitação no Compras.gov")
    parser.add_argument("--menu", action="store_true", help="Usar menu interativo")
    parser.add_argument("--tipo", choices=["licitacao", "item"], help="Tipo de consulta")
    parser.add_argument("--modo", choices=["completos", "todos"], default="completos", help="Modo de retorno")
    parser.add_argument("--output-format", choices=["csv", "json"], default="csv", help="Formato de saída")
    parser.add_argument("--output", help="Nome do arquivo de saída")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Número máximo de páginas a consultar. Se omitido, não há limite.")
    parser.add_argument("--uasg", type=int, help="Filtro de UASG")
    parser.add_argument("--modalidade", type=int, help="Filtro de modalidade")
    parser.add_argument("--codigo-item-material", type=int, help="Código do item de material")
    parser.add_argument("--codigo-item-servico", type=int, help="Código do item de serviço")
    parser.add_argument("--cnpj-fornecedor", type=str, help="CNPJ do fornecedor")
    parser.add_argument("--cpf-vencedor", type=str, help="CPF do vencedor")
    parser.add_argument("--data-publicacao-inicial", type=str, help="Data de publicação inicial YYYY-MM-DD")
    parser.add_argument("--data-publicacao-final", type=str, help="Data de publicação final YYYY-MM-DD")
    parser.add_argument("--pertence14133", action="store_true", help="Filtrar somente licitações da Lei 14.133/2021")
    return parser


def validar_filtros_licitacao(filtros):
    """Valida filtros obrigatórios para consulta de licitações."""
    if not any(filtros.get(key) is not None for key in ["uasg", "modalidade"]):
        raise ValueError("Para licitação, informe ao menos uasg, modalidade.")
    if not filtros.get("data_publicacao_inicial") or not filtros.get("data_publicacao_final"):
        raise ValueError("Para licitação, informe data_publicacao_inicial e data_publicacao_final.")


def validar_filtros_item(filtros):
    """Valida filtros obrigatórios para consulta de itens de licitação."""
    if not any(
        filtros.get(key) is not None
        for key in [
            "uasg",
            "modalidade",
            "codigo_item_material",
            "codigo_item_servico",
            "cnpj_fornecedor",
            "cpfVencedor",
        ]
    ):
        raise ValueError("Para itens, informe ao menos um filtro: uasg, modalidade, codigo_item_material, codigo_item_servico, cnpj_fornecedor ou cpfVencedor.")


def montar_filtros(args):
    """Transforma argumentos de CLI em um dicionário de filtros para a API."""
    filtros = {}
    for campo, valor in [
        ("uasg", args.uasg),
        ("modalidade", args.modalidade),
        ("codigo_item_material", args.codigo_item_material),
        ("codigo_item_servico", args.codigo_item_servico),
        ("cnpj_fornecedor", args.cnpj_fornecedor),
        ("cpfVencedor", args.cpf_vencedor),
        ("data_publicacao_inicial", args.data_publicacao_inicial),
        ("data_publicacao_final", args.data_publicacao_final),
    ]:
        if valor is not None:
            filtros[campo] = valor
    if args.pertence14133:
        filtros["pertence14133"] = True
    return filtros


def executar_consulta_com_args(args):
    """Executa consulta quando a ferramenta é chamada pela linha de comando."""
    filtros = montar_filtros(args)
    if args.tipo == "licitacao":
        validar_filtros_licitacao(filtros)
        resultados, metadata = consultar_licitacoes(filtros, args.modo, args.max_pages)
        imprimir_licitacoes(resultados, metadata, args.modo)
    else:
        validar_filtros_item(filtros)
        resultados, metadata = consultar_itens_licitacao(filtros, args.modo, args.max_pages)
        imprimir_itens(resultados, metadata, args.modo)

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = gerar_nome_arquivo(args.tipo, args.output_format, filtros)

    if args.output_format == "csv":
        if args.tipo == "licitacao":
            salvar_csv(output_file, resultados, ["uasg", "modalidade", "nome_modalidade", "objeto", "data_publicacao", "situacao"])
        else:
            salvar_csv(output_file, resultados, ["numero_item_licitacao", "codigo_item_material", "codigo_item_servico", "descricao_item", "quantidade", "valor_estimado", "cnpj_fornecedor"])
    else:
        salvar_json(output_file, resultados)


def menu():
    """Menu interativo para choises rápidas e importação de filtros."""
    print("API Compras.gov - Licitações e Itens de Licitações (SALC / 12º GAAAe)")
    while True:
        print("\nMenu:")
        print("1 - Consultar licitações")
        print("2 - Consultar itens de licitações")
        print("0 - Sair")
        opcao = input("Digite a opção: ").strip()

        if opcao == "1":
            filtros = solicitar_filtros_licitacao()
            modo = input("Modo de consulta [completos/todos]: ").strip().lower()
            if modo not in {"completos", "todos"}:
                modo = "completos"
            max_pages = pedir_input("max_pages", int, DEFAULT_MAX_PAGES, obrigatorio=False)
            resultados, metadata = consultar_licitacoes(filtros, modo, max_pages)
            imprimir_licitacoes(resultados, metadata, modo)
            if resultados:
                salvar = input("Salvar em CSV/JSON? (csv/json/n): ").strip().lower()
                if salvar in {"csv", "json"}:
                    nome = input("Nome do arquivo [deixe em branco para gerar automático]: ").strip()
                    arquivo = Path(nome) if nome else gerar_nome_arquivo("licitacao", salvar, filtros)
                    if salvar == "csv":
                        salvar_csv(arquivo, resultados, ["uasg", "modalidade", "nome_modalidade", "objeto", "data_publicacao", "situacao"])
                    else:
                        salvar_json(arquivo, resultados)

        elif opcao == "2":
            filtros = solicitar_filtros_item()
            modo = input("Modo de consulta [completos/todos]: ").strip().lower()
            if modo not in {"completos", "todos"}:
                modo = "completos"
            max_pages = pedir_input("max_pages", int, DEFAULT_MAX_PAGES, obrigatorio=False)
            resultados, metadata = consultar_itens_licitacao(filtros, modo, max_pages)
            imprimir_itens(resultados, metadata, modo)
            if resultados and PANDAS_AVAILABLE:
                df_resultados = registros_para_dataframe(resultados)
                
                detectar_anomalias_scipy(df_resultados)
                
                ver_grafico = input("\nDeseja visualizar os gráficos com Seaborn? (s/n): ").strip().lower()
                if ver_grafico in {'s', 'sim', 'y'}:
                    visualizar_resultados_seaborn(df_resultados)
            if resultados:
                salvar = input("Salvar em CSV/JSON? (csv/json/n): ").strip().lower()
                if salvar in {"csv", "json"}:
                    nome = input("Nome do arquivo [deixe em branco para gerar automático]: ").strip()
                    arquivo = Path(nome) if nome else gerar_nome_arquivo("item", salvar, filtros)
                    if salvar == "csv":
                        salvar_csv(arquivo, resultados, ["numero_item_licitacao", "codigo_item_material", "codigo_item_servico", "descricao_item", "quantidade", "valor_estimado", "cnpj_fornecedor"])
                    else:
                        salvar_json(arquivo, resultados)

        elif opcao == "0":
            print("Encerrando.")
            break
        else:
            print("Opção inválida.")



def main():
    """Ponto de entrada principal que escolhe entre modo interativo ou CLI."""
    parser = criar_parser()
    args = parser.parse_args()

    if args.menu or len(sys.argv) == 1:
        menu()
        return

    if not args.tipo:
        parser.error("--tipo é obrigatório quando não se usa --menu")

    try:
        executar_consulta_com_args(args)
    except ValueError as erro:
        print(f"Entrada inválida: {erro}")
        sys.exit(1)
    except RuntimeError as erro:
        print(f"Erro de requisição: {erro}")
        sys.exit(1)


if __name__ == "__main__":
    main()
