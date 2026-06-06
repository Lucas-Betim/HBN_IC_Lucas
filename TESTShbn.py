from dataclasses import dataclass

from HNB import HNB
from HBNLearner import ParameterLearner
from HBNpipeline import (
    build_and_train,
    get_candidate_latents_from_original_nb
)

# =========================
# Testes para HNB
# =========================

# =========================
# Helpers de relatório
# =========================


@dataclass
class TestResult:
    name: str
    status: str          # PASS | FAIL | XFAIL
    detail: str = ""


class TestRunner:
    def __init__(self):
        self.results: list[TestResult] = []

    def add_pass(self, name: str, detail: str = ""):
        self.results.append(TestResult(
            name=name, status="PASS", detail=detail))

    def add_fail(self, name: str, detail: str = ""):
        self.results.append(TestResult(
            name=name, status="FAIL", detail=detail))

    def add_xfail(self, name: str, detail: str = ""):
        """Expected failure (falha esperada)."""
        self.results.append(TestResult(
            name=name, status="XFAIL", detail=detail))

    def summary(self):
        total = len(self.results)
        passed = sum(r.status == "PASS" for r in self.results)
        failed = sum(r.status == "FAIL" for r in self.results)
        xfailed = sum(r.status == "XFAIL" for r in self.results)

        print("\n" + "=" * 60)
        print("RESUMO DOS TESTES")
        print("=" * 60)
        for r in self.results:
            print(f"{r.status:5} | {r.name} | {r.detail}")

        print("-" * 60)
        print(
            f"TOTAL: {total} | PASS: {passed} | FAIL: {failed} | XFAIL (esperado): {xfailed}")
        print("=" * 60)

# =========================
# Testes
# =========================


def test_1_positive_case(runner: TestRunner, csv_file: str):
    print("\n=== TESTE 1: caso correto (deve passar, exceto regularidade) ===")
    model, df_final = build_and_train(csv_file)

    # is_tree
    try:
        val = model.is_tree()
        if val is True:
            runner.add_pass("is_tree() no caso correto", "True")
        else:
            runner.add_fail("is_tree() no caso correto",
                            f"Esperado True, veio {val}")
    except Exception as e:
        runner.add_fail("is_tree() no caso correto",
                        f"{type(e).__name__}: {e}")

    # isRegular (pode dar False dependendo da regra)
    try:
        val = model.isRegular()
        # Aqui não forço PASS/FAIL rígido, porque sua regra ainda é "pseudo"
        runner.add_pass("isRegular() executa sem quebrar", f"retornou {val}")
    except Exception as e:
        runner.add_fail("isRegular() executa sem quebrar",
                        f"{type(e).__name__}: {e}")

    # check_model (pode falhar por regularidade)
    try:
        model.check_model()
        runner.add_pass("check_model() no caso correto", "OK")
    except Exception as e:
        # Se falhou por regularidade, marcamos como XFAIL (falha esperada por definição atual)
        msg = str(e).lower()
        if "regular" in msg:
            runner.add_xfail("check_model() no caso correto",
                             f"Falhou por regularidade: {type(e).__name__}: {e}")
        else:
            runner.add_fail("check_model() no caso correto",
                            f"{type(e).__name__}: {e}")


def test_2_tree_violation_two_parents(runner: TestRunner, csv_file: str):
    print("\n=== TESTE 2: violação de árvore (2 pais) ===")
    model, df_final = build_and_train(csv_file)

    # Força 2 pais para 'precip': já tem HL1->precip, adiciona class->precip
    try:
        model.add_edge("class", "precip")
        runner.add_pass("forçar 2 pais (add_edge class->precip)",
                        "aresta adicionada")
    except Exception as e:
        runner.add_fail("forçar 2 pais (add_edge class->precip)",
                        f"{type(e).__name__}: {e}")
        return

    # is_tree deve dar False
    try:
        val = model.is_tree()
        if val is False:
            runner.add_pass("is_tree() detecta 2 pais", "False (correto)")
        else:
            runner.add_fail("is_tree() detecta 2 pais",
                            f"Esperado False, veio {val}")
    except Exception as e:
        runner.add_fail("is_tree() detecta 2 pais", f"{type(e).__name__}: {e}")

    # check_model deve falhar (pode falhar por CPD inconsistente antes)
    try:
        model.check_model()
        runner.add_fail("check_model() com 2 pais",
                        "Era para falhar, mas passou")
    except Exception as e:
        runner.add_pass("check_model() com 2 pais falha",
                        f"{type(e).__name__}: {e}")


def test_3_cycle_violation(runner: TestRunner, csv_file: str):
    print("\n=== TESTE 3: ciclo (pgmpy deve bloquear no add_edge) ===")
    model, df_final = build_and_train(csv_file)

    # Já existe HL1->temp, tenta temp->HL1
    try:
        model.add_edge("temp", "HL1")
        runner.add_fail("add_edge(temp->HL1) cria ciclo",
                        "Era para dar erro, mas não deu")
    except Exception as e:
        # Isso é o comportamento esperado: pgmpy impede loops.
        runner.add_pass("add_edge(temp->HL1) bloqueado",
                        f"{type(e).__name__}: {e}")

    # O modelo ainda deve estar válido estruturalmente (exceto regularidade)
    try:
        model.check_model()
        runner.add_pass("check_model() após tentativa de ciclo", "OK")
    except Exception as e:
        msg = str(e).lower()
        if "regular" in msg:
            runner.add_xfail("check_model() após tentativa de ciclo",
                             f"Falhou por regularidade: {type(e).__name__}: {e}")
        else:
            runner.add_fail("check_model() após tentativa de ciclo",
                            f"{type(e).__name__}: {e}")


def test_4_missing_cpd(runner: TestRunner, csv_file: str):
    print("\n=== TESTE 4: CPD faltando (deve falhar) ===")
    model, df_final = build_and_train(csv_file)

    try:
        cpd_temp = model.get_cpds("temp")
        model.remove_cpds(cpd_temp)
        runner.add_pass("remover CPD(temp)", "CPD removida")
    except Exception as e:
        runner.add_fail("remover CPD(temp)", f"{type(e).__name__}: {e}")
        return

    try:
        model.check_model()
        runner.add_fail("check_model() com CPD faltando",
                        "Era para falhar, mas passou")
    except Exception as e:
        runner.add_pass("check_model() com CPD faltando falha",
                        f"{type(e).__name__}: {e}")


def test_5_can_insert_latent(runner: TestRunner, csv_file: str):
    print("\n=== TESTE 5: can_insert_latent ===")
    model, df_final = build_and_train(csv_file)

    # Depois de inserir HL1, precip/temp/wind são filhos de HL1, não de class.
    try:
        val = model.can_insert_latent("class", ["precip", "temp", "wind"])
        if val is False:
            runner.add_pass("can_insert_latent(class, filhos)",
                            "False (correto)")
        else:
            runner.add_fail("can_insert_latent(class, filhos)",
                            f"Esperado False, veio {val}")
    except Exception as e:
        runner.add_fail("can_insert_latent(class, filhos)",
                        f"{type(e).__name__}: {e}")

    try:
        val = model.can_insert_latent("HL1", ["precip", "temp", "wind"])
        if val is True:
            runner.add_pass("can_insert_latent(HL1, filhos)", "True (correto)")
        else:
            runner.add_fail("can_insert_latent(HL1, filhos)",
                            f"Esperado True, veio {val}")
    except Exception as e:
        runner.add_fail("can_insert_latent(HL1, filhos)",
                        f"{type(e).__name__}: {e}")
