import pandas as pd
from pgmpy.readwrite import BIFWriter


def gerar_relatorios_hbn(
    model,
    history: list,
    initial_score: float,
    final_score: float,
    output_prefix: str = "hbn",
    bootstrap_result: dict | None = None,
    run_config: dict | None = None
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

    if bootstrap_result is not None:
        resultado.update({
            "versao_avaliacao": bootstrap_result["evaluation_version"],
            "protocolo_avaliacao": "holdout_bootstrap",
            "acuracia_holdout": bootstrap_result["accuracy_holdout"],
            "acuracia_media_bootstrap": bootstrap_result["accuracy_mean"],
            "desvio_padrao_acuracia_bootstrap": bootstrap_result["accuracy_std"],
            "numero_repeticoes_bootstrap": bootstrap_result["n_bootstrap"],
            "proporcao_teste": bootstrap_result["test_size"],
            "numero_amostras_teste": bootstrap_result["n_test"]
        })

    if run_config is not None:
        resultado.update(run_config)

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

    arquivo_bootstrap = None
    if bootstrap_result is not None:
        arquivo_bootstrap = f"bootstrap_{output_prefix}.csv"
        bootstrap_result["bootstrap_results"].to_csv(
            arquivo_bootstrap, index=False
        )

    # =========================
    # 3. Exportar modelo BIF
    # =========================

    arquivo_bif = f"modelo_final_{output_prefix}.bif"
    writer = BIFWriter(model)
    writer.write_bif(arquivo_bif)

    return {
        "resultado_csv": arquivo_resultado,
        "historico_csv": arquivo_historico,
        "bootstrap_csv": arquivo_bootstrap,
        "avaliacao_csv": arquivo_bootstrap,
        "modelo_bif": arquivo_bif
    }
