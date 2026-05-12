"""
Heuristic subtype classifiers for EGE tasks 6, 10, 12, 13 and 16.

Each task has a fixed set of subtypes that drive personalised retrieval
and conspect generation.  ``classify_subtype`` dispatches to the
per-task classifier.
"""

from __future__ import annotations

import re

SUBTYPES_10: tuple[str, ...] = (
    "percent_mixture",
    "equation_setup",
    "straight_motion",
    "circular_motion",
    "water_motion",
    "joint_work",
    "progression",
    "other",
)

SUBTYPES_12: tuple[str, ...] = (
    "no_derivative",
    "power_irrational",
    "quotient",
    "product",
    "exp_log",
    "trig",
    "other",
)

SUBTYPES_6: tuple[str, ...] = (
    "exponential_eq",
    "logarithmic_eq",
    "trigonometric_eq",
    "irrational_eq",
    "quadratic_eq",
    "combined_eq",
    "other",
)

SUBTYPES_13: tuple[str, ...] = (
    "simplest_trig",
    "quadratic_in_one_func",
    "homogeneous_first",
    "homogeneous_second",
    "non_homogeneous_first",
    "factorisation",
    "fractional_zero_numerator",
    "exp_trig",
    "log_trig",
    "mixed_double_angle",
    "interval_selection",
    "other",
)

SUBTYPES_16: tuple[str, ...] = (
    "annuity_credit",
    "differentiated_credit",
    "debt_schedule_credit",
    "find_rate_or_term",
    "deposit",
    "optimisation",
    "other",
)

# Human-readable labels for conspect prompts and mistake summaries (Russian).
SUBTYPE_LABELS: dict[str, str] = {
    "joint_work": "совместную работу",
    "water_motion": "движение по воде",
    "straight_motion": "прямолинейное движение",
    "circular_motion": "движение по окружности",
    "percent_mixture": "проценты и смеси",
    "equation_setup": "составление уравнений",
    "progression": "прогрессии",
    "no_derivative": "анализ графика без производной",
    "power_irrational": "степенные и иррациональные функции",
    "quotient": "дробные функции",
    "product": "произведение функций",
    "exp_log": "показательные и логарифмические функции",
    "trig": "тригонометрические функции",
    "exponential_eq": "показательные уравнения",
    "logarithmic_eq": "логарифмические уравнения",
    "trigonometric_eq": "тригонометрические уравнения",
    "irrational_eq": "иррациональные уравнения",
    "quadratic_eq": "квадратные и приводимые к ним уравнения",
    "combined_eq": "смешанные уравнения и замены",
    # Task 13
    "simplest_trig": "простейшие тригонометрические уравнения",
    "quadratic_in_one_func": "уравнения, сводимые к квадратному",
    "homogeneous_first": "однородные тригонометрические уравнения первой степени",
    "homogeneous_second": "однородные тригонометрические уравнения второй степени",
    "non_homogeneous_first": "линейные тригонометрические уравнения вида a·sin x + b·cos x = c",
    "factorisation": "уравнения, решаемые разложением на множители",
    "fractional_zero_numerator": "дробно-рациональные тригонометрические уравнения",
    "exp_trig": "показательно-тригонометрические уравнения",
    "log_trig": "логарифмо-тригонометрические уравнения",
    "mixed_double_angle": "уравнения со смешанными формулами (двойной угол, приведение)",
    "interval_selection": "отбор корней на отрезке",
    # Task 16
    "annuity_credit": "кредит с равными платежами (аннуитет)",
    "differentiated_credit": "кредит с равными частями тела долга (дифференцированный)",
    "debt_schedule_credit": "кредит с заданным графиком долга",
    "find_rate_or_term": "обратная задача (найти ставку, срок или максимальную сумму)",
    "deposit": "вклад с пополнениями",
    "optimisation": "оптимизация прибыли или издержек",
    "other": "задачу",
}


def normalize_text(s: str) -> str:
    """Lower-case and collapse whitespace for keyword matching."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def classify_subtype_10(text: str) -> str:
    """Classify a task-10 problem text into one of ``SUBTYPES_10``."""
    t = normalize_text(text)

    if any(
        k in t
        for k in ("прогресс", "арифметическ", "геометрическ", "a_n", "s_n", "q^", "член прогрессии")
    ):
        return "progression"

    if any(
        k in t
        for k in (
            "труба",
            "насос",
            "резервуар",
            "бак",
            "выполнит",
            "детал",
            "работая вместе",
            "производительность",
        )
    ):
        return "joint_work"

    has_current = any(k in t for k in ("течение", "по течению", "против течения"))
    has_boat = any(k in t for k in ("плот", "катер", "лодк", "баржа", "теплоход", "пароход"))
    is_lake = any(k in t for k in ("озер", "пруд", "водохранилищ"))
    if has_current and not is_lake:
        return "water_motion"
    if has_boat and is_lake:
        return "straight_motion"
    if has_current:
        return "water_motion"
    if has_boat:
        return "water_motion"

    if any(k in t for k in ("круг", "окружност", "кольц", "стрелк", "часов")):
        return "circular_motion"

    if any(k in t for k in ("выехал", "встрет", "км/ч", "скорост", "прибыл", "догнал", "обгон")):
        return "straight_motion"

    if any(k in t for k in ("сплав", "раствор", "смес", "концентрац", "процент", "%", "доля")):
        return "percent_mixture"

    if any(k in t for k in ("уравнен", "неизвест", "составьте", "найдите значение")):
        return "equation_setup"

    return "other"


def classify_subtype_6(text: str) -> str:
    """Classify a task-6 problem text into one of ``SUBTYPES_6``."""
    t = normalize_text(text)

    if any(
        k in t
        for k in (
            "sin",
            "cos",
            "tg",
            "ctg",
            "триг",
            "синус",
            "косинус",
            "тангенс",
            "арксинус",
            "arcsin",
        )
    ):
        return "trigonometric_eq"

    if any(k in t for k in ("log", "ln", "логарифм")):
        return "logarithmic_eq"

    if any(k in t for k in ("корень", "sqrt", "иррацион")):
        return "irrational_eq"

    if any(k in t for k in ("^", "степен", "показател", "a^x", "e^", "2^x", "3^x", "5^x")):
        return "exponential_eq"

    if any(k in t for k in ("квадратн", "дискриминант", "биквадрат", "x^2", "x²")):
        return "quadratic_eq"

    if any(k in t for k in ("модул", "|", "систем", "уравнен")):
        return "combined_eq"

    return "other"


def classify_subtype_12(text: str) -> str:
    """Classify a task-12 problem text into one of ``SUBTYPES_12``."""
    t = normalize_text(text)

    if any(k in t for k in ("sin", "cos", "tg", "ctg", "триг", "синус", "косинус", "тангенс")):
        return "trig"

    if any(k in t for k in ("ln", "log", "логарифм", "e^", "exp", "экспонент")):
        return "exp_log"

    if "/" in t and ("x" in t or "f(" in t):
        return "quotient"
    if "·" in t or "*" in t or ("(" in t and ")(" in t):
        return "product"

    if any(k in t for k in ("корень", "sqrt", "x^", "степен", "многочлен", "производн")):
        return "power_irrational"

    if any(k in t for k in ("график", "по рисунку", "без производной", "возрастает", "убывает")):
        return "no_derivative"

    return "other"


def classify_subtype_13(text: str) -> str:
    """Classify a task-13 problem text into one of ``SUBTYPES_13``."""
    t = normalize_text(text)

    has_trig = any(k in t for k in ("sin", "cos", "tg", "ctg", "син", "кос", "танг", "котанг"))

    has_log = any(k in t for k in ("log", "ln ", "ln(", "лог"))
    has_exp = any(k in t for k in ("^x", "^{x", "^\\sin", "^{\\sin", "^\\cos", "^{\\cos"))
    has_frac_zero = ("\\frac" in t or "/" in t) and ("=0" in t.replace(" ", "") or "= 0" in t)

    if has_log and has_trig:
        return "log_trig"
    if has_exp and has_trig:
        return "exp_trig"
    if has_frac_zero and has_trig:
        return "fractional_zero_numerator"

    has_double_angle = any(k in t for k in ("2x", "2 x", "\\cos2", "\\sin2", "cos2x", "sin2x"))
    has_squared = any(k in t for k in ("^2", "²", "{2}"))

    if "tg" in t and has_squared:
        return "quadratic_in_one_func"

    if has_double_angle and has_trig:
        return "mixed_double_angle"

    if has_squared and has_trig:
        if (
            "sin" in t
            and "cos" in t
            and any(k in t for k in ("\\sin x\\cos", "sinxcos", "sin x cos"))
        ):
            return "homogeneous_second"
        return "quadratic_in_one_func"

    if "sin" in t and "cos" in t and not has_squared:
        return "non_homogeneous_first"

    if has_trig:
        return "simplest_trig"

    return "other"


def classify_subtype_16(text: str) -> str:
    """Classify a task-16 problem text into one of ``SUBTYPES_16``."""
    t = normalize_text(text)

    has_credit = any(k in t for k in ("кредит", "взять в долг", "погасить", "заёмщик", "заемщик"))
    has_deposit = any(k in t for k in ("вклад", "вкладчик", "депозит", "положил", "пополн"))

    has_table = any(k in t for k in ("таблиц", "доля", "следующей таблице", "должен составлять"))
    has_equal_payments = any(
        k in t for k in ("равные плате", "одинаковые сумм", "равных плате", "ежегодно по")
    )
    has_diff = any(
        k in t
        for k in (
            "одну и ту же сумму меньше",
            "одну и ту же величину меньше",
            "равными частями",
            "равными долями",
        )
    )
    has_find_rate_term = any(
        k in t
        for k in (
            "найдите r",
            "чему равно r",
            "ставк",
            "найдите n",
            "найдите наименьш",
            "целое число лет",
            "сколько лет",
        )
    )
    has_optim = any(k in t for k in ("прибыл", "издержк", "себестоимост", "произвед"))

    if has_optim and not has_credit and not has_deposit:
        return "optimisation"

    if has_deposit and not has_credit:
        return "deposit"

    if has_table:
        return "debt_schedule_credit"

    if has_diff:
        return "differentiated_credit"

    if has_equal_payments:
        return "annuity_credit"

    if has_find_rate_term:
        return "find_rate_or_term"

    if has_credit:
        return "annuity_credit"

    return "other"


def classify_subtype(text: str, task_number: int = 10) -> str:
    """Dispatch to the per-task classifier.

    Args:
        text: Raw task prompt text.
        task_number: EGE task number (6, 10, 12, 13, or 16).

    Returns:
        Subtype string from the corresponding ``SUBTYPES_*`` tuple.
    """
    if task_number == 6:
        return classify_subtype_6(text)
    if task_number == 12:
        return classify_subtype_12(text)
    if task_number == 13:
        return classify_subtype_13(text)
    if task_number == 16:
        return classify_subtype_16(text)
    return classify_subtype_10(text)
