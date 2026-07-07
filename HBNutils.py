import os
import pandas as pd


def get_base_name(csv_file: str) -> str:
    """
    Retorna o nome da base sem pasta e sem extensão.
    """
    return os.path.splitext(os.path.basename(csv_file))[0]


def detect_class_column(df: pd.DataFrame) -> str:
    """
    Detecta automaticamente a coluna de classe.

    Prioridade:
    1) nomes comuns: class, target, label, y, output
    2) se não encontrar, usa a última coluna
    """

    candidates = [
        "class", "Class", "CLASS",
        "target", "Target", "TARGET",
        "label", "Label", "LABEL",
        "y", "Y",
        "output", "Output", "OUTPUT"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return df.columns[-1]


def choose_class_column(csv_file: str) -> str:
    """
    Mostra a coluna detectada automaticamente e permite o usuário confirmar
    ou digitar outro nome.
    """

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    detected = detect_class_column(df)

    print("\nColunas encontradas:")
    for i, col in enumerate(df.columns, start=1):
        marker = "  <-- detectada" if col == detected else ""
        print(f"{i} - {col}{marker}")

    resposta = input(
        f"\nColuna da classe [ENTER = {detected}]: "
    ).strip()

    if resposta == "":
        return detected

    if resposta not in df.columns:
        raise ValueError(
            f"Coluna '{resposta}' não existe no CSV. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    return resposta


def print_section(title: str):
    """
    Imprime cabeçalho padronizado no terminal.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def escolher_csv_em_pasta(default_folder: str = "bases") -> str:
    """
    Lista arquivos CSV de uma pasta e permite o usuário escolher.
    Se a pasta não existir ou estiver vazia, permite digitar o caminho manualmente.
    """

    pasta = input(f"Pasta das bases [ENTER = {default_folder}]: ").strip()

    if pasta == "":
        pasta = default_folder

    if not os.path.isabs(pasta):
        pasta = os.path.join(os.path.dirname(__file__), pasta)

    if not os.path.exists(pasta):
        print(f"[AVISO] Pasta não encontrada: {pasta}")
        caminho = input("Digite o caminho do CSV manualmente: ").strip()

        if not os.path.isabs(caminho):
            caminho = os.path.join(os.path.dirname(__file__), caminho)

        if not os.path.exists(caminho):
            raise FileNotFoundError(f"CSV não encontrado: {caminho}")

        return caminho

    arquivos_csv = [
        f for f in os.listdir(pasta)
        if f.lower().endswith(".csv")
    ]

    if len(arquivos_csv) == 0:
        print(f"[AVISO] Nenhum CSV encontrado em: {pasta}")
        caminho = input("Digite o caminho do CSV manualmente: ").strip()

        if not os.path.isabs(caminho):
            caminho = os.path.join(os.path.dirname(__file__), caminho)

        if not os.path.exists(caminho):
            raise FileNotFoundError(f"CSV não encontrado: {caminho}")

        return caminho

    print("\nBases disponíveis:")
    for i, arquivo in enumerate(arquivos_csv, start=1):
        print(f"{i} - {arquivo}")

    escolha = input("\nEscolha o número da base: ").strip()

    if not escolha.isdigit():
        raise ValueError("Escolha inválida. Digite um número.")

    idx = int(escolha)

    if idx < 1 or idx > len(arquivos_csv):
        raise ValueError("Número fora da lista de bases.")

    return os.path.join(pasta, arquivos_csv[idx - 1])