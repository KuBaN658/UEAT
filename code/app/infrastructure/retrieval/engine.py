"""
Hybrid RAG retrieval engine.

Pipeline per query
------------------
1. Embed query (dense + sparse).
2. Hybrid Qdrant search (RRF fusion, optional subtypes filter).
3. Multi-query RRF fusion (RAG-Fusion over multiple query strings).
4. Profile boost: ``error_score`` + ``skill_score`` per atom.
5. Mastery penalty: down-rank atoms the student likely mastered.
6. Prerequisite-graph expansion (PSI-KT bidirectional).
7. MMR diversity re-ranking.
8. Topological sort for coherent output ordering.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass

from qdrant_client import models as qmodels

from app.core.config import get_rag_settings
from app.domain.analysis import atom_weakness, mastery_level
from app.domain.atom import Atom
from app.domain.atoms import (
    ATOM_BY_ID,
    ATOMS,
    get_dependents,
    get_prerequisites,
    topological_order,
    validate_error_tags,
)
from app.domain.profile import StudentProfile
from app.domain.subtypes import SUBTYPE_LABELS
from app.infrastructure.vector.embeddings import (
    embed_query,
    embed_query_colbert,
    embed_query_sparse,
    embed_texts,
    embed_texts_colbert,
    embed_texts_sparse,
)
from app.infrastructure.vector.qdrant_store import IndexedDoc, QdrantStore

log = logging.getLogger(__name__)


# ── Human-readable error descriptions ───────────────────────────────

_ERROR_HINTS: dict[str, str] = {
    # Task 10
    "river_swap_plus_minus": "Путаешь формулы по/против течения: по течению скорость $v+u$, против — $v-u$",
    "river_forgot_v_gt_u": "Забываешь проверить, что собственная скорость катера больше скорости течения ($v > u$)",
    "work_added_times_instead_of_rates": "Складываешь времена вместо производительностей: надо $\\frac{1}{t_1}+\\frac{1}{t_2}$, а не $t_1+t_2$",
    "work_wrong_rate_equation": "Неверно составляешь уравнение совместной работы",
    "motion_wrong_relative_speed": "Ошибаешься со скоростью сближения или удаления",
    "motion_forgot_delay_or_stop": "Пропускаешь задержку старта или остановку при подсчёте общего времени",
    "motion_mixed_time_and_distance": "Путаешь время и расстояние в формулах",
    "avg_speed_arithmetic_mean": "Считаешь среднюю скорость как среднее арифметическое вместо $S_{\\text{общ}}/T_{\\text{общ}}$",
    "avg_speed_used_wrong_total_time": "Неправильно находишь общее время или общее расстояние",
    "percent_used_as_number": "Используешь процент как число, забывая делить на 100",
    "percent_wrong_base": "Берёшь процент от неправильной базы (не от того числа)",
    "percent_chain_changes": "При последовательных изменениях складываешь проценты вместо умножения множителей",
    "mixture_not_using_balance": "Не составляешь баланс вещества при смешивании",
    "mixture_percent_not_to_fraction": "Забываешь перевести проценты в доли (делить на 100) при составлении уравнения",
    "mixture_wrong_total_mass": "Ошибаешься с общей массой при смешивании",
    "picked_wrong_variable": "Неудачно выбираешь переменную — уравнение получается сложнее, чем нужно",
    "lost_parentheses_or_sign": "Теряешь скобки или знак при алгебраических преобразованиях",
    "did_not_check_constraints": "Не проверяешь ограничения: скорость > 0, время > 0, масса > 0",
    "units_mismatch": "Единицы измерения не совпадают (часы и минуты, км и м)",
    "progression_wrong_formula": "Путаешь формулы арифметической и геометрической прогрессий",
    "progression_off_by_one": "Ошибаешься на единицу: $n$ вместо $n-1$ в формуле прогрессии",
    # Task 12
    "deriv_wrong_rule": "Неверно берёшь производную (ошибка в правиле дифференцирования)",
    "deriv_forgot_chain_rule": "Пропускаешь цепное правило: забываешь умножить на производную внутренней функции",
    "deriv_sign_error": "Теряешь знак минус в производной",
    "critical_point_wrong_eq": "Неверно решаешь уравнение $f'(x) = 0$",
    "critical_point_missed_domain": "Не учитываешь область определения (например, $\\ln x$ при $x > 0$)",
    "minmax_forgot_endpoints": "Забываешь проверить значения функции на концах отрезка",
    "minmax_wrong_sign_analysis": "Ошибаешься в анализе знака производной на интервалах",
    "minmax_confused_min_max": "Путаешь точку экстремума (значение $x$) и значение функции $f(x)$",
    "trig_deriv_error": "Ошибаешься в производных $\\sin$/$\\cos$ (знак минус у косинуса!)",
    "log_deriv_error": "Ошибаешься в производной логарифма",
    "exp_deriv_error": "Ошибаешься в производной экспоненты",
    "quotient_rule_error": "Ошибка в формуле производной дроби (знак в числителе)",
    "product_rule_error": "Ошибка в формуле производной произведения (забываешь второе слагаемое)",
    # Task 6
    "wrong_base_reduction": "Неверно приводишь показательное уравнение к общему основанию или ошибаешься в степени",
    "log_domain_missed": "Не учитываешь ОДЗ логарифма (аргумент должен быть $>0$)",
    "log_property_error": "Неверно применяешь свойства логарифмов (сумма/разность/степень)",
    "exp_property_error": "Неверно применяешь свойства степеней при преобразовании уравнения",
    "trig_lost_solutions": "Теряешь решения тригонометрического уравнения (периодичность, серии)",
    "trig_extra_solutions": "Включаешь лишние корни после преобразований (проверь подстановкой)",
    "irrational_no_check": "Не проверяешь корни иррационального уравнения после возведения в степень",
    "quadratic_discriminant_error": "Ошибка в дискриминанте или в формуле корней квадратного уравнения",
    "forgot_back_substitution": "Решил замену $t$, но не вернулся к исходной переменной $x$",
    "sign_error_in_equation": "Ошибка знака при переносе слагаемых или раскрытии скобок",
    "unknown": "Не удалось определить причину ошибки (сервис диагностики недоступен).",
}


def _human_error(tag: str) -> str:
    """Return a human-readable Russian description for an error tag."""
    return _ERROR_HINTS.get(tag, tag.replace("_", " "))


# ── LLM output sanitisation ──────────────────────────────────────────

_TECH_RE = re.compile(
    r"(?i)"
    r"(?:t(?:06|1[02])_mm\d{1,2})"
    r"|(?:phase|scaffold|mastery|micromodule|prereq|subtype)"
    r"|(?:buggy.?rule|stepwise|diagnos)"
    r"|(?:фаза\s*\d)"
    r"|(?:порядок закрытия пробелов)"
)

_BRACKET_ID_RE = re.compile(r"\(t(?:06|1[02])_mm\d{1,2}(?:,\s*t(?:06|1[02])_mm\d{1,2})*\)")

_META_CORRECTED_RE = re.compile(
    r"^\s*(?:В конспекте обнаружены ошибки|Ниже приведены исправленные версии|Исправленный конспект готов)\b",
    flags=re.IGNORECASE,
)


def sanitize_llm_output(text: str) -> str:
    """Remove leaked internal identifiers and ``<think>`` blocks from LLM responses."""
    if not text or not str(text).strip():
        return text

    def _apply_line_rules(body: str) -> str:
        lines = body.split("\n")
        clean: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                clean.append(line)
                continue
            if _META_CORRECTED_RE.search(stripped):
                continue
            if _TECH_RE.search(stripped) and len(stripped) < 120:
                cleaned = _TECH_RE.sub("", stripped).strip(" .,:-–—•*#")
                if len(cleaned) < 15:
                    continue
            clean.append(line)
        return "\n".join(clean)

    original = text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = _BRACKET_ID_RE.sub("", text)
    text = re.sub(r"t(?:06|1[02])_mm\d{1,2}", "", text)
    out = _apply_line_rules(text)
    if out.strip():
        return out
    tail = re.split(r"</think>\s*", original, flags=re.IGNORECASE | re.MULTILINE)
    if len(tail) > 1:
        after = tail[-1].strip()
        if after:
            return _apply_line_rules(after)
    return original.strip()


# ── Internal helpers ─────────────────────────────────────────────────


def _atoms_content_hash(atoms: list[Atom]) -> str:
    h = hashlib.sha256()
    for a in atoms:
        h.update(f"{a.id}|{a.text}\n".encode())
    return h.hexdigest()[:16]


def _sparse_to_qdrant(se: object) -> qmodels.SparseVector:
    """Convert a fastembed ``SparseEmbedding`` to a Qdrant ``SparseVector``."""
    return qmodels.SparseVector(
        indices=se.indices.tolist(),  # type: ignore[attr-defined]
        values=se.values.tolist(),  # type: ignore[attr-defined]
    )


@dataclass(frozen=True)
class Retrieved:
    """A single retrieval result with its final score."""

    atom: Atom
    score: float


def _rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion (Cormack et al., SIGIR 2009)."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return dict(scores)


def _subtype_sim(a: Atom, b: Atom) -> float:
    """Jaccard similarity of subtypes, used for MMR diversity."""
    sa, sb = set(a.subtypes), set(b.subtypes)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _mmr_select(candidates: list[tuple[str, float]], k: int, lambda_: float = 0.6) -> list[str]:
    """Maximal Marginal Relevance: select *k* diverse atoms from scored candidates."""
    if len(candidates) <= k:
        return [aid for aid, _ in candidates]
    scores = dict(candidates)
    min_s, max_s = min(scores.values()), max(scores.values())
    rng = max_s - min_s if max_s > min_s else 1.0

    selected: list[str] = []
    remaining = [aid for aid, _ in candidates]
    for _ in range(k):
        best_aid = None
        best_mmr = float("-inf")
        for aid in remaining:
            if aid in selected:
                continue
            atom = ATOM_BY_ID.get(aid)
            if not atom:
                continue
            rel = (scores[aid] - min_s) / rng
            max_sim = 0.0
            for sid in selected:
                other = ATOM_BY_ID.get(sid)
                if other:
                    max_sim = max(max_sim, _subtype_sim(atom, other))
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_aid = aid
        if best_aid is None:
            break
        selected.append(best_aid)
        remaining = [x for x in remaining if x != best_aid]
    return selected


def _extract_concrete_mistakes(
    profile: StudentProfile, task_number: int | None = None
) -> list[str]:
    """Extract 1–2 concrete mistake examples from the error-event log."""
    out: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for ev in reversed(profile.error_events[-20:]):
        tn = ev.get("task_number")
        if task_number is not None and tn != task_number:
            continue
        sa = ev.get("student_answer")
        ca = ev.get("correct_answer")
        if sa is None or ca is None:
            continue
        subtype = ev.get("subtype", "other")
        key = (str(sa), str(ca), subtype)
        if key in seen:
            continue
        seen.add(key)
        label = SUBTYPE_LABELS.get(subtype, subtype.replace("_", " "))
        out.append(f"В задаче на {label} ответил {sa}, правильный ответ {ca}")
        if len(out) >= 2:
            break
    return out


# ── Conspect prompt builders ─────────────────────────────────────────

_CONSPECT_DIVERSITY_HINT = (
    "Разнообразие примеров: не копируй дословно числа из справочного блока как единственный разбор. "
    "Избегай шаблонного примера f(x)=x³−3x²+2 на отрезке [1;4] и теплохода 200 км / 15 км/ч / стоянка 10 ч — "
    "подбирай другие коэффициенты, отрезки [a;b], контексты №10 (работа, проценты, движение). "
    "Для №6 варьируй тип уравнения: показательное, логарифмическое, тригонометрическое, с корнем, квадратное — не только одно и то же. "
    "Для №12 варьируй тип функции: многочлен, ln, e^x, дробь, тригонометрия — не только кубический многочлен."
)


def _conspect_practice_focus_lines(profile: StudentProfile) -> str:
    """Bias «Потренируйся» toward subtypes where the student has wrong attempts."""
    pairs = [(st, profile.wrong.get(st, 0)) for st in profile.wrong]
    pairs = [(st, c) for st, c in pairs if c > 0]
    pairs.sort(key=lambda x: -x[1])
    if not pairs:
        return ""
    labels: list[str] = []
    for st, _ in pairs[:4]:
        labels.append(SUBTYPE_LABELS.get(st, st.replace("_", " ")))
    return (
        f"В разделе «Потренируйся»: учти подтипы, где ученик чаще ошибался: {', '.join(labels)}. "
        "Сбалансируй: для №6 — тип уравнения по слабым местам; для №10 — контекст (движение / работа / проценты / река); "
        "для №12 — max/min на отрезке по соответствующим подтипам."
    )


def _conspect_build_errors_block(
    profile: StudentProfile,
    human_errors: list[str],
    top_errors_list: list[str],
    recent_wrong: list[dict] | None,
    concrete_mistakes: list[str] | None,
) -> str:
    total_att = sum(profile.attempts.values())
    errors_block = ""
    if human_errors:
        items = "\n".join(f"  - {e}" for e in human_errors)
        errors_block = f"Ученик чаще всего допускает такие ошибки:\n{items}"
        if recent_wrong:
            wrong_items = []
            for i, w in enumerate(recent_wrong[:3], 1):
                tt = (w.get("task_text") or "").strip()[:400]
                sa = w.get("student_answer", "?")
                ca = w.get("correct_answer", "?")
                tn = w.get("task_number", "")
                wrong_items.append(
                    f"Ошибка {i} (задание {tn}):\n"
                    f"Условие: {tt}\n"
                    f"Ответ ученика: {sa}\n"
                    f"Правильный ответ: {ca}"
                )
            errors_block += (
                "\n\nФактические ошибки ученика (используй для целевых объяснений):\n"
                + "\n\n".join(wrong_items)
            )
        elif concrete_mistakes:
            errors_block += "\n\nКонкретные примеры:\n" + "\n".join(
                f"  - {m}" for m in concrete_mistakes[:2]
            )
        if not recent_wrong and profile.error_events:
            seen_tt: set[str] = set()
            lines: list[str] = []
            for ev in reversed(profile.error_events[-24:]):
                tt = (ev.get("task_text") or "").strip()
                if len(tt) < 20:
                    continue
                key = tt[:400]
                if key in seen_tt:
                    continue
                seen_tt.add(key)
                tn = ev.get("task_number", "")
                lines.append(f"  - (задание {tn}): {tt[:380]}")
                if len(lines) >= 5:
                    break
            if lines:
                errors_block += (
                    "\n\nКонтекст прошлых неверных решений (текст с платформы; ответа ученика нет):\n"
                    + "\n".join(lines)
                )
        if (
            top_errors_list
            and top_errors_list[0] == "picked_wrong_variable"
            and sum(profile.wrong.values()) > 0
        ):
            errors_block += (
                "\n\nВажно по заданию 10: по архиву часто не видно подтипа текстовой задачи — дай сбалансированный обзор "
                "основных типов №10 (движение, совместная работа, проценты/смеси, прогрессии) и типовых ловушек при выборе переменной."
            )
    else:
        errors_block = "У ученика пока нет статистики ошибок — дай общий конспект по теме."
        if total_att > 0:
            parts = sorted(profile.attempts.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
            sub = ", ".join(f"{k} ({v})" for k, v in parts)
            errors_block += (
                f"\n\nУже были попытки по подтипам: {sub} — включи это в примеры и чеклист."
            )
    return errors_block


# ── Main engine ──────────────────────────────────────────────────────


class RagEngine:
    """Hybrid retrieval engine backed by a Qdrant vector store.

    Initialised once at application startup (inside ``lifespan``).
    Validates error tags, embeds all atoms, and populates Qdrant if needed.
    """

    def __init__(self) -> None:
        validate_error_tags()
        self._atoms = ATOMS
        cfg = get_rag_settings()

        texts = [self._atom_to_doc_text(a) for a in self._atoms]
        content_hash = _atoms_content_hash(self._atoms)

        dense_vecs = embed_texts(texts)
        dim = dense_vecs.shape[1]
        self._store = QdrantStore(dim=dim)

        if self._store.is_populated(
            len(self._atoms),
            content_hash,
            bm25_enabled=cfg.bm25_enabled,
            colbert_enabled=cfg.colbert_enabled,
        ):
            log.info(
                "Qdrant collection up-to-date (%d points), skipping re-embedding",
                len(self._atoms),
            )
        else:
            log.info("Populating Qdrant (%d atoms)...", len(self._atoms))

            sparse_vecs = None
            if cfg.bm25_enabled:
                sparse_embs = embed_texts_sparse(texts)
                sparse_vecs = [_sparse_to_qdrant(se) for se in sparse_embs]

            colbert_vecs = None
            colbert_dim = None
            if cfg.colbert_enabled:
                log.info("Computing ColBERT embeddings for %d atoms...", len(texts))
                colbert_vecs = embed_texts_colbert(texts)
                colbert_dim = colbert_vecs[0].shape[1] if colbert_vecs else None

            docs = [
                IndexedDoc(
                    doc_id=a.id,
                    text=texts[i],
                    meta={
                        "subtypes": list(a.subtypes),
                        "error_tags": list(a.error_tags),
                        "title": a.title,
                        "task_number": a.task_number,
                    },
                )
                for i, a in enumerate(self._atoms)
            ]
            self._store.ensure_collection(bm25_enabled=cfg.bm25_enabled, colbert_dim=colbert_dim)
            self._store.upsert_batch(
                docs, dense_vecs, sparse_vecs, content_hash, colbert_vecs=colbert_vecs
            )

    @staticmethod
    def _atom_to_doc_text(a: Atom) -> str:
        return f"{a.title}\n{a.text}"

    def retrieve(
        self,
        *,
        query: str | list[str],
        task_number: int | None,
        subtype: str | None,
        profile: StudentProfile,
        k: int = 12,
    ) -> list[Retrieved]:
        """Retrieve the top-*k* atoms for the given query and student profile.

        Steps: multi-query hybrid search → RRF fusion → profile boost →
        mastery penalty → graph expansion → MMR selection → topological sort.

        Args:
            query: Single query string or a list of query strings (multi-query).
            task_number: Filter atoms by EGE task number, or ``None`` for all.
            subtype: Filter by subtype, or ``None`` / ``"all_subtypes"`` for all.
            profile: The student profile used for personalisation.
            k: Number of atoms to return.

        Returns:
            Topologically ordered list of ``Retrieved`` objects.
        """
        cfg = get_rag_settings()
        queries = [query] if isinstance(query, str) else query
        if not queries:
            q_text = (
                f"ЕГЭ профильная математика задание {task_number} обзор"
                if task_number
                else "ЕГЭ профильная математика обзор"
            )
            queries = [q_text]

        sub_filter = [subtype] if subtype and subtype != "all_subtypes" else None
        fetch_limit = max(48, k * 4)

        ranked_lists: list[list[str]] = []
        for q in queries:
            q_vec = embed_query(q)
            q_sparse = _sparse_to_qdrant(embed_query_sparse(q)) if cfg.bm25_enabled else None
            q_colbert = embed_query_colbert(q) if cfg.colbert_enabled else None
            hits = self._store.hybrid_search(
                q_vec,
                q_sparse,
                query_text=q if cfg.reranker_enabled else None,
                colbert_vec=q_colbert,
                task_number=task_number,
                subtypes=sub_filter,
                limit=fetch_limit,
                prefetch_factor=cfg.qdrant_prefetch_factor,
                dense_weight=cfg.dense_weight,
                bm25_weight=cfg.bm25_weight,
                colbert_weight=cfg.colbert_weight,
                rrf_k=cfg.rrf_k,
            )
            ranked_lists.append([h.doc.doc_id for h in hits])

        rrf_scores = _rrf_fuse(ranked_lists, k=cfg.rrf_k)
        scores: dict[str, float] = dict(rrf_scores)

        gamma = cfg.profile_weight
        for aid in list(scores.keys()):
            atom = ATOM_BY_ID.get(aid)
            if not atom:
                continue
            err_score = sum(profile.error_score(tag) for tag in atom.error_tags)
            skill_score = sum(profile.skill_scores.get(skill, 0.0) for skill in atom.shared_skills)
            scores[aid] += gamma * (err_score + cfg.skill_factor * skill_score)
            m = mastery_level(profile, aid)
            if m > cfg.mastery_threshold:
                scores[aid] -= cfg.mastery_penalty

        base_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[: max(8, k)]
        for aid in base_ids:
            atom = ATOM_BY_ID.get(aid)
            if not atom or (task_number is not None and atom.task_number != task_number):
                continue
            for p in get_prerequisites(aid, depth=2):
                prereq_factor = (
                    cfg.cross_task_prereq_bonus
                    if task_number is not None and p.task_number != task_number
                    else 1.0
                )
                weakness = atom_weakness(profile, p.id)
                bonus = cfg.prereq_bonus * prereq_factor * (
                    1.0 + weakness
                ) + cfg.prereq_error_tag_bonus * max(0, len(p.error_tags))
                scores[p.id] = max(scores.get(p.id, float("-inf")), scores[aid] + bonus)
            for dep in get_dependents(aid):
                if task_number is not None and dep.task_number != task_number:
                    continue
                if subtype and subtype != "all_subtypes" and subtype not in dep.subtypes:
                    continue
                scores[dep.id] = max(
                    scores.get(dep.id, float("-inf")), scores[aid] + cfg.dependent_bonus
                )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        candidates = ranked[: k * 2]
        by_id = {aid: (ATOM_BY_ID[aid], sc) for aid, sc in candidates if aid in ATOM_BY_ID}
        selected_ids = _mmr_select(candidates, k, cfg.mmr_lambda)
        selected_atoms = [ATOM_BY_ID[aid] for aid in selected_ids if aid in ATOM_BY_ID]
        ordered = topological_order(selected_atoms)
        return [Retrieved(atom=a, score=by_id[a.id][1]) for a in ordered if a.id in by_id][:k]

    def build_personalized_conspect_prompt(
        self,
        *,
        task_number: int | None = None,
        subtype: str,
        profile: StudentProfile,
        retrieved: list[Retrieved],
        frontier_atoms: list[str] | None = None,
        concrete_mistakes: list[str] | None = None,
        recent_wrong: list[dict] | None = None,
    ) -> dict[str, str | int]:
        """Build the personalised conspect context dict (no LLM call).

        Returns a dict with keys ``task_scope``, ``detail``, ``errors_block``,
        ``frontier_block``, ``diversity_block``, ``context_section``,
        ``n_err_items``, ``top_err``.
        """
        from app.domain.analysis import top_errors

        top_errors_list = top_errors(profile, 8, None if subtype == "all_subtypes" else subtype)
        human_errors = [_human_error(t) for t in top_errors_list[:5]]
        n_errors = len(human_errors)

        content_blocks: list[str] = []
        misconception_blocks: list[str] = []
        for i, r in enumerate(retrieved, 1):
            block = f"{i}. {r.atom.title}\n{r.atom.text}"
            if r.atom.id.startswith("mc_"):
                misconception_blocks.append(block)
            else:
                content_blocks.append(block)

        cfg = get_rag_settings()
        total_att = sum(profile.attempts.values())
        total_wrong = sum(profile.wrong.values())
        accuracy = 0.0 if total_att == 0 else (total_att - total_wrong) / total_att
        if accuracy < cfg.accuracy_low:
            detail = "подробный (все шаги расписаны, каждая формула объяснена)"
        elif accuracy < cfg.accuracy_mid:
            detail = "средний (ключевые шаги подробно, остальное коротко)"
        else:
            detail = "краткий (формулы + напоминания + акцент на ошибках)"

        errors_block = _conspect_build_errors_block(
            profile, human_errors, top_errors_list, recent_wrong, concrete_mistakes
        )

        practice_focus = _conspect_practice_focus_lines(profile)
        diversity_block = _CONSPECT_DIVERSITY_HINT
        if practice_focus:
            diversity_block = practice_focus + "\n\n" + diversity_block

        frontier_block = ""
        if frontier_atoms:
            frontier_block = (
                "\n\nСледующие темы для изучения (слабо освоены, но пререквизиты готовы):\n"
                + "\n".join(f"  - {t}" for t in frontier_atoms[:3])
            )

        context_parts: list[str] = []
        if misconception_blocks:
            context_parts.append(
                "Известные типичные заблуждения (обязательно учти в разделе «Типичные ошибки» и «Найди ошибку»):\n"
                + "\n\n".join(misconception_blocks)
            )
        context_parts.append(
            "Справочный материал (используй для содержания, но НЕ копируй дословно и НЕ упоминай номера/названия):\n"
            + "\n\n".join(content_blocks)
            if content_blocks
            else "Нет дополнительного справочного материала."
        )
        context_section = "\n\n" + "---\n\n".join(context_parts)

        task_scope = f"заданию №{task_number}" if task_number else "заданиям №6, №10 и №12"
        top_err = human_errors[0] if human_errors else "типичная ошибка"
        n_err_items = n_errors if n_errors > 0 else 2

        return {
            "task_scope": task_scope,
            "detail": detail,
            "errors_block": errors_block,
            "frontier_block": frontier_block,
            "diversity_block": diversity_block,
            "context_section": context_section,
            "n_err_items": n_err_items,
            "top_err": top_err,
        }
