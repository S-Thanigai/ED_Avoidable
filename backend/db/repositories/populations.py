"""
populations.py
---------------
Repository for saved populations and everything under them (members,
ED visits, care records, optional safety context, and the persisted
analysis results). Every read/write here takes an explicit
`owner_user_id` argument that the CALLER must have already derived from
the authenticated session (backend/auth.py's get_current_user) -- this
module never trusts a caller-supplied owner id, and every query filters
on it, so a population/member belonging to a different user simply does
not match and is reported as not-found (never a 403 that would confirm
existence to an attacker).

Reload/search/pagination/filter functions (list_members_paginated,
get_member_detail, get_population_summary) read ONLY the already-computed
analysis_results table -- none of them call the ML pipeline. The only
place this module ever writes an analysis_results row is
create_population_with_analysis(), which the router calls exactly once,
with decisions already produced by the SAME UC07Orchestrator used by
POST /uc07/decide (see backend/routers/populations.py).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Select, func, insert, select
from sqlalchemy.orm import Session

from ..decision_codec import analysis_result_to_decision_dict, decision_dict_to_analysis_result_kwargs
from ..models import (
    AnalysisResult,
    Population,
    PopulationCareRecord,
    PopulationEdVisit,
    PopulationMember,
    PopulationSafetyContext,
)


def _native(value):
    """numpy/pandas scalar -> plain Python value, NaN/NaT -> None. pyodbc
    executemany needs native types, not numpy scalars."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _as_date(value) -> dt.date:
    return pd.Timestamp(value).date()


# ---------------------------------------------------------------------------
# Listing / metadata / deletion
# ---------------------------------------------------------------------------


def list_populations(db: Session, owner_user_id: int) -> list[Population]:
    stmt = (
        select(Population)
        .where(Population.owner_user_id == owner_user_id)
        .order_by(Population.created_at.desc())
    )
    return list(db.execute(stmt).scalars().all())


def get_owned_population(db: Session, owner_user_id: int, population_id: int) -> Population | None:
    stmt = select(Population).where(
        Population.id == population_id, Population.owner_user_id == owner_user_id
    )
    return db.execute(stmt).scalar_one_or_none()


def delete_population(db: Session, owner_user_id: int, population_id: int) -> bool:
    population = get_owned_population(db, owner_user_id, population_id)
    if population is None:
        return False
    db.delete(population)  # cascades to members/ed_visits/care/safety/analysis_results
    db.commit()
    return True


# Percent-of-probability bin edges, matching
# frontend/src/uc07/components/AnalyticsCharts.tsx's BIN_EDGES exactly
# (0/10/20/30/40/50, last bin open-ended) so a saved population's
# probability histogram renders identical bins to the live CSV pathway's
# client-computed one -- just aggregated server-side instead of requiring
# every member's row to be sent to the browser.
_PROBABILITY_BIN_EDGES = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)


def get_population_summary(db: Session, owner_user_id: int, population_id: int) -> dict | None:
    """Aggregate-only summary (tier/safety/navigation/probability-bin
    counts via SQL GROUP BY) -- never fetches per-member rows. Safe to
    call for a population with tens of thousands of members."""
    population = get_owned_population(db, owner_user_id, population_id)
    if population is None:
        return None

    tier_rows = db.execute(
        select(AnalysisResult.tier, func.count())
        .where(AnalysisResult.population_id == population_id)
        .group_by(AnalysisResult.tier)
    ).all()
    safety_rows = db.execute(
        select(AnalysisResult.safety_state, func.count())
        .where(AnalysisResult.population_id == population_id)
        .group_by(AnalysisResult.safety_state)
    ).all()
    navigation_rows = db.execute(
        select(AnalysisResult.navigation_destination, func.count())
        .where(AnalysisResult.population_id == population_id)
        .group_by(AnalysisResult.navigation_destination)
    ).all()

    # NOTE: a single `GROUP BY CASE WHEN ...` query was tried here first
    # and rejected by SQL Server/pyodbc with "column ... is invalid in
    # the select list because it is not contained in either an aggregate
    # function or the GROUP BY clause" -- a known mssql+pyodbc quirk
    # where the driver's parameterization of the CASE's bound literals
    # keeps SQL Server from proving the SELECT and GROUP BY expressions
    # are identical, even though they are. Six discrete range-count
    # queries sidesteps it entirely and is just as index-friendly (each
    # is filtered by population_id first).
    probability_bins: list[int] = []
    for i in range(6):
        lower = _PROBABILITY_BIN_EDGES[i]
        stmt = select(func.count()).where(
            AnalysisResult.population_id == population_id, AnalysisResult.probability >= lower
        )
        if i < 5:
            upper = _PROBABILITY_BIN_EDGES[i + 1]
            stmt = stmt.where(AnalysisResult.probability < upper)
        probability_bins.append(db.execute(stmt).scalar_one())

    thresholds_row = db.execute(
        select(AnalysisResult.moderate_threshold, AnalysisResult.high_threshold)
        .where(AnalysisResult.population_id == population_id)
        .limit(1)
    ).first()

    return {
        "population": population,
        "tier_counts": {tier: count for tier, count in tier_rows},
        "safety_counts": {state: count for state, count in safety_rows},
        "navigation_counts": {(dest or "NONE"): count for dest, count in navigation_rows},
        "probability_bins": probability_bins,
        "moderate_threshold": thresholds_row[0] if thresholds_row else None,
        "high_threshold": thresholds_row[1] if thresholds_row else None,
    }


# ---------------------------------------------------------------------------
# Transactional save (called once, at "Save to Database" time)
# ---------------------------------------------------------------------------


@dataclass
class SaveAnalysisInput:
    owner_user_id: int
    name: str
    index_date: dt.date
    model_version: str
    dataset_id: str
    synthetic_model: bool
    source_members_filename: str | None
    source_ed_visits_filename: str | None
    source_care_filename: str | None
    source_safety_context_filename: str | None
    members_df: pd.DataFrame
    ed_df: pd.DataFrame
    care_df: pd.DataFrame
    safety_context_df: pd.DataFrame | None
    # member_id -> {red_flag, icu, admitted, major_procedure, triage_level} (each Optional[int])
    safety_contexts: dict[str, dict]
    # decision_to_dict() output per member (backend/agents/orchestrator.py)
    decisions: list[dict]


def create_population_with_analysis(db: Session, data: SaveAnalysisInput) -> Population:
    """One DB transaction: population row + batch inserts for members,
    ED visits, care records, optional safety context, and analysis
    results. Any failure rolls back the whole thing -- no half-imported
    population is ever left behind."""
    try:
        population = Population(
            owner_user_id=data.owner_user_id,
            name=data.name,
            source_members_filename=data.source_members_filename,
            source_ed_visits_filename=data.source_ed_visits_filename,
            source_care_filename=data.source_care_filename,
            source_safety_context_filename=data.source_safety_context_filename,
            index_date=data.index_date,
            model_version=data.model_version,
            dataset_id=data.dataset_id,
            synthetic_model=data.synthetic_model,
            member_count=len(data.decisions),
        )
        db.add(population)
        db.flush()  # assigns population.id without committing yet

        pid = population.id

        member_records = [
            {
                "population_id": pid,
                "member_id": str(row.member_id),
                "age": _native(row.age),
                "gender": _native(row.gender),
                "diabetes": _native(row.diabetes),
                "copd": _native(row.copd),
                "hypertension": _native(row.hypertension),
                "chf": _native(row.chf),
                "asthma": _native(row.asthma),
                "ckd": _native(row.ckd),
                "num_chronic_conditions": _native(row.num_chronic_conditions),
                "transportation_barrier": _native(row.transportation_barrier),
                "telehealth_available": _native(row.telehealth_available),
                "pcp_distance_miles": _native(row.pcp_distance_miles),
                "urgent_care_distance_miles": _native(row.urgent_care_distance_miles),
            }
            for row in data.members_df.itertuples(index=False)
        ]
        if member_records:
            db.execute(insert(PopulationMember), member_records)

        ed_records = [
            {
                "population_id": pid,
                "visit_id": _native(getattr(row, "visit_id", None)),
                "member_id": str(row.member_id),
                "visit_date": _as_date(row.visit_date),
                "diagnosis": _native(getattr(row, "diagnosis", None)),
                "triage_level": _native(row.triage_level),
                "admitted": _native(row.admitted),
                "icu": _native(row.icu),
                "major_procedure": _native(row.major_procedure),
                "cost": _native(row.cost),
                "red_flag": _native(row.red_flag),
            }
            for row in data.ed_df.itertuples(index=False)
        ]
        if ed_records:
            db.execute(insert(PopulationEdVisit), ed_records)

        care_records = [
            {
                "population_id": pid,
                "care_id": _native(getattr(row, "care_id", None)),
                "member_id": str(row.member_id),
                "visit_date": _as_date(row.visit_date),
                "care_type": _native(row.care_type),
            }
            for row in data.care_df.itertuples(index=False)
        ]
        if care_records:
            db.execute(insert(PopulationCareRecord), care_records)

        if data.safety_contexts:
            safety_records = [
                {
                    "population_id": pid,
                    "member_id": member_id,
                    "red_flag": ctx.get("red_flag"),
                    "icu": ctx.get("icu"),
                    "admitted": ctx.get("admitted"),
                    "major_procedure": ctx.get("major_procedure"),
                    "triage_level": ctx.get("triage_level"),
                }
                for member_id, ctx in data.safety_contexts.items()
            ]
            if safety_records:
                db.execute(insert(PopulationSafetyContext), safety_records)

        analysis_records = [
            decision_dict_to_analysis_result_kwargs(pid, decision) for decision in data.decisions
        ]
        if analysis_records:
            db.execute(insert(AnalysisResult), analysis_records)

        db.commit()
        db.refresh(population)
        return population
    except Exception:
        db.rollback()
        raise


# ---------------------------------------------------------------------------
# Paginated / searchable member reads (never rerun the model)
# ---------------------------------------------------------------------------

SORTABLE_COLUMNS: dict[str, "object"] = {
    "member_id": AnalysisResult.member_id,
    "tier": AnalysisResult.tier,
    "probability": AnalysisResult.probability,
    "navigation": AnalysisResult.navigation_destination,
}


def _base_member_query(population_id: int) -> Select:
    return select(AnalysisResult).where(AnalysisResult.population_id == population_id)


def list_members_paginated(
    db: Session,
    owner_user_id: int,
    population_id: int,
    *,
    page: int = 1,
    page_size: int = 15,
    search: str | None = None,
    tier: str | None = None,
    navigation: str | None = None,
    safety: str | None = None,
    prob_min: float | None = None,
    prob_max: float | None = None,
    sort_key: str | None = None,
    sort_dir: str = "asc",
) -> dict | None:
    population = get_owned_population(db, owner_user_id, population_id)
    if population is None:
        return None

    stmt = _base_member_query(population_id)
    if search:
        stmt = stmt.where(AnalysisResult.member_id.ilike(f"%{search.strip()}%"))
    if tier and tier != "ALL":
        stmt = stmt.where(AnalysisResult.tier == tier)
    if navigation and navigation != "ALL":
        if navigation == "NONE":
            stmt = stmt.where(AnalysisResult.navigation_destination.is_(None))
        else:
            stmt = stmt.where(AnalysisResult.navigation_destination == navigation)
    if safety and safety != "ALL":
        stmt = stmt.where(AnalysisResult.safety_state == safety)
    if prob_min is not None:
        stmt = stmt.where(AnalysisResult.probability >= prob_min)
    if prob_max is not None:
        stmt = stmt.where(AnalysisResult.probability <= prob_max)

    total_items = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    total_pages = max(1, math.ceil(total_items / page_size))
    page = max(1, min(page, total_pages))

    order_column = SORTABLE_COLUMNS.get(sort_key or "", AnalysisResult.member_id)
    order_clause = order_column.desc() if sort_dir == "desc" else order_column.asc()
    stmt = stmt.order_by(order_clause).offset((page - 1) * page_size).limit(page_size)

    rows = db.execute(stmt).scalars().all()
    items = [analysis_result_to_decision_dict(row) for row in rows]

    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
    }


def get_member_detail(db: Session, owner_user_id: int, population_id: int, member_id: str) -> dict | None:
    population = get_owned_population(db, owner_user_id, population_id)
    if population is None:
        return None

    result_row = db.execute(
        select(AnalysisResult).where(
            AnalysisResult.population_id == population_id, AnalysisResult.member_id == member_id
        )
    ).scalar_one_or_none()
    if result_row is None:
        return None

    profile_row = db.execute(
        select(PopulationMember).where(
            PopulationMember.population_id == population_id, PopulationMember.member_id == member_id
        )
    ).scalar_one_or_none()

    ed_rows = (
        db.execute(
            select(PopulationEdVisit)
            .where(PopulationEdVisit.population_id == population_id, PopulationEdVisit.member_id == member_id)
            .order_by(PopulationEdVisit.visit_date.desc())
        )
        .scalars()
        .all()
    )
    care_rows = (
        db.execute(
            select(PopulationCareRecord)
            .where(PopulationCareRecord.population_id == population_id, PopulationCareRecord.member_id == member_id)
            .order_by(PopulationCareRecord.visit_date.desc())
        )
        .scalars()
        .all()
    )
    safety_context_row = db.execute(
        select(PopulationSafetyContext).where(
            PopulationSafetyContext.population_id == population_id,
            PopulationSafetyContext.member_id == member_id,
        )
    ).scalar_one_or_none()

    return {
        "decision": analysis_result_to_decision_dict(result_row),
        "profile": None
        if profile_row is None
        else {
            "member_id": profile_row.member_id,
            "age": profile_row.age,
            "gender": profile_row.gender,
            "diabetes": profile_row.diabetes,
            "copd": profile_row.copd,
            "hypertension": profile_row.hypertension,
            "chf": profile_row.chf,
            "asthma": profile_row.asthma,
            "ckd": profile_row.ckd,
            "num_chronic_conditions": profile_row.num_chronic_conditions,
            "transportation_barrier": profile_row.transportation_barrier,
            "telehealth_available": profile_row.telehealth_available,
            "pcp_distance_miles": profile_row.pcp_distance_miles,
            "urgent_care_distance_miles": profile_row.urgent_care_distance_miles,
        },
        "ed_visits": [
            {
                "visit_id": r.visit_id,
                "member_id": r.member_id,
                "visit_date": r.visit_date.isoformat(),
                "diagnosis": r.diagnosis,
                "triage_level": r.triage_level,
                "admitted": r.admitted,
                "icu": r.icu,
                "major_procedure": r.major_procedure,
                "cost": r.cost,
                "red_flag": r.red_flag,
            }
            for r in ed_rows
        ],
        "care_visits": [
            {
                "care_id": r.care_id,
                "member_id": r.member_id,
                "visit_date": r.visit_date.isoformat(),
                "care_type": r.care_type,
            }
            for r in care_rows
        ],
        "safety_context_captured_at": (
            safety_context_row.captured_at.isoformat() if safety_context_row is not None else None
        ),
    }
