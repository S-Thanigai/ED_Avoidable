"""
routers/populations.py
------------------------
Saved-population endpoints: list, metadata+summary, delete, save-analysis,
paginated/searchable members, and single-member detail.

Every endpoint depends on `auth.get_current_user` and passes
`current_user.id` as the ONLY source of ownership to the repository
layer (backend/db/repositories/populations.py) -- never a client-
supplied owner id. A population/member that exists but belongs to a
different user is indistinguishable, from the response, from one that
doesn't exist at all (404) -- this avoids confirming existence to an
unauthorized caller.

POST /populations/save-analysis is the only endpoint that touches the
ML pipeline, and it does so by calling the SAME shared helpers
`POST /uc07/decide` uses (backend/uc07_pipeline.py) -- never a second
prediction implementation, never trusting client-supplied
probability/tier/safety values.
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
from db.engine import get_db
from db.models import User
from db.repositories import populations as populations_repo
from db.repositories.populations import SaveAnalysisInput
from orchestrator import decision_to_dict
from uc07_pipeline import (
    decide_for_population,
    get_uc07_orchestrator,
    parse_index_date,
    read_csv,
    resolve_safety_contexts,
    validate_and_parse_uc07_inputs,
)

router = APIRouter(prefix="/populations", tags=["Populations"])


# ---------------------------------------------------------------------------
# List / metadata / delete
# ---------------------------------------------------------------------------


class PopulationSummary(BaseModel):
    id: int
    name: str
    member_count: int
    index_date: date
    model_version: str
    dataset_id: str
    synthetic_model: bool
    created_at: str
    updated_at: str


def _to_population_summary(population) -> PopulationSummary:
    return PopulationSummary(
        id=population.id,
        name=population.name,
        member_count=population.member_count,
        index_date=population.index_date,
        model_version=population.model_version,
        dataset_id=population.dataset_id,
        synthetic_model=population.synthetic_model,
        created_at=population.created_at.isoformat(),
        updated_at=population.updated_at.isoformat(),
    )


@router.get("", response_model=list[PopulationSummary])
def list_populations(
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> list[PopulationSummary]:
    populations = populations_repo.list_populations(db, current_user.id)
    return [_to_population_summary(p) for p in populations]


@router.get("/{population_id}")
def get_population(
    population_id: int,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    summary = populations_repo.get_population_summary(db, current_user.id, population_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Population not found.")
    return {
        **_to_population_summary(summary["population"]).model_dump(),
        "tier_counts": summary["tier_counts"],
        "safety_counts": summary["safety_counts"],
        "navigation_counts": summary["navigation_counts"],
        "probability_bins": summary["probability_bins"],
        "moderate_threshold": summary["moderate_threshold"],
        "high_threshold": summary["high_threshold"],
    }


@router.delete("/{population_id}", status_code=204)
def delete_population(
    population_id: int,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    deleted = populations_repo.delete_population(db, current_user.id, population_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Population not found.")
    return None


# ---------------------------------------------------------------------------
# Save analysis -- recomputes via the SAME orchestrator path /uc07/decide
# uses, then persists the result in one transaction.
# ---------------------------------------------------------------------------


@router.post("/save-analysis", response_model=PopulationSummary, status_code=201)
async def save_analysis(
    name: str = Form(..., min_length=1, max_length=200),
    index_date: str | None = Form(None),
    current_safety_context: str | None = Form(None),
    members_file: UploadFile = File(...),
    ed_visits_file: UploadFile = File(...),
    care_file: UploadFile = File(...),
    safety_context_file: UploadFile | None = File(None),
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> PopulationSummary:
    orchestrator = get_uc07_orchestrator()

    members_df = read_csv(members_file)
    ed_df = read_csv(ed_visits_file)
    care_df = read_csv(care_file)

    members_df, ed_df = validate_and_parse_uc07_inputs(members_df, ed_df, care_df)

    parsed_index_date = parse_index_date(index_date)

    safety_context_df = read_csv(safety_context_file) if safety_context_file is not None else None
    contexts = resolve_safety_contexts(members_df, current_safety_context, safety_context_df)

    decisions = decide_for_population(
        orchestrator, members_df, ed_df, care_df, parsed_index_date, contexts, member_id=None,
    )
    if not decisions:
        raise HTTPException(status_code=422, detail="No members found to analyze in the uploaded data.")

    decision_dicts = [decision_to_dict(d) for d in decisions]
    safety_context_payload = {
        member_id: {
            "red_flag": ctx.red_flag,
            "icu": ctx.icu,
            "admitted": ctx.admitted,
            "major_procedure": ctx.major_procedure,
            "triage_level": ctx.triage_level,
        }
        for member_id, ctx in contexts.items()
        if ctx.provided
    }

    save_input = SaveAnalysisInput(
        owner_user_id=current_user.id,
        name=name.strip(),
        index_date=parsed_index_date,
        model_version=orchestrator.risk_agent.model_version,
        dataset_id=orchestrator.risk_agent.dataset_id,
        synthetic_model=orchestrator.risk_agent.synthetic_model,
        source_members_filename=members_file.filename,
        source_ed_visits_filename=ed_visits_file.filename,
        source_care_filename=care_file.filename,
        source_safety_context_filename=safety_context_file.filename if safety_context_file else None,
        members_df=members_df,
        ed_df=ed_df,
        care_df=care_df,
        safety_context_df=safety_context_df,
        safety_contexts=safety_context_payload,
        decisions=decision_dicts,
    )

    try:
        population = populations_repo.create_population_with_analysis(db, save_input)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Saving the analyzed population failed: {exc}") from exc

    return _to_population_summary(population)


# ---------------------------------------------------------------------------
# Paginated / searchable members -- pure reads, never rerun the model.
# ---------------------------------------------------------------------------


class PaginatedMembers(BaseModel):
    items: list[dict]
    page: int
    page_size: int
    total_items: int
    total_pages: int


@router.get("/{population_id}/members", response_model=PaginatedMembers)
def list_members(
    population_id: int,
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
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> PaginatedMembers:
    page = max(1, page)
    page_size = max(1, min(page_size, 200))  # a client can't force an unbounded fetch
    result = populations_repo.list_members_paginated(
        db,
        current_user.id,
        population_id,
        page=page,
        page_size=page_size,
        search=search,
        tier=tier,
        navigation=navigation,
        safety=safety,
        prob_min=prob_min,
        prob_max=prob_max,
        sort_key=sort_key,
        sort_dir=sort_dir,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Population not found.")
    return PaginatedMembers(**result)


@router.get("/{population_id}/members/{member_id}")
def get_member(
    population_id: int,
    member_id: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    detail = populations_repo.get_member_detail(db, current_user.id, population_id, member_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Member not found in this population.")
    return detail
