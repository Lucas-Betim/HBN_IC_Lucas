import pandas as pd
from pgmpy.readwrite import BIFWriter


def gerar_relatorios_hbn(
    model,
    history: list,
    initial_score: float,
    final_score: float,
    output_prefix: str = "hbn"
):
    """
    Gera arquivos finais da execução HBN:
      - resultado_hbn.csv
      - historico_hbn.csv
      - modelo_final.bif
    """

    # =========================
    # 1. Resultado resumido
    # =========================

    latentes_finais = list(model.latents) if hasattr(model, "latents") else []

    resultado = {
        "score_inicial": initial_score,
        "score_final": final_score,
        "ganho_score": final_score - initial_score,
        "numero_latentes_finais": len(latentes_finais),
        "latentes_finais": "; ".join(latentes_finais)
    }

    df_resultado = pd.DataFrame([resultado])
    arquivo_resultado = f"resultado_{output_prefix}.csv"
    df_resultado.to_csv(arquivo_resultado, index=False)

    # =========================
    # 2. Histórico das iterações
    # =========================

    linhas_historico = []

    for h in history:
        linhas_historico.append({
            "iteracao": h.get("iteration"),
            "aceita": h.get("accepted"),
            "latente": h.get("latent_name", ""),
            "par": str(h.get("pair", "")),
            "score": h.get("score", h.get("best_candidate_score")),
            "cardinalidade_inicial": h.get("initial_cardinality", ""),
            "cardinalidade_final": h.get("latent_cardinality", ""),
            "numero_colapsos": len(h.get("collapse_history", [])),
            "score_atual": h.get("current_score", "")
        })

    df_historico = pd.DataFrame(linhas_historico)
    arquivo_historico = f"historico_{output_prefix}.csv"
    df_historico.to_csv(arquivo_historico, index=False)

    # =========================
    # 3. Exportar modelo BIF
    # =========================

    arquivo_bif = f"modelo_final_{output_prefix}.bif"
    writer = BIFWriter(model)
    writer.write_bif(arquivo_bif)

    return {
        "resultado_csv": arquivo_resultado,
        "historico_csv": arquivo_historico,
        "modelo_bif": arquivo_bif
    }