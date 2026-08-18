"""
models.py
---------
SQLAlchemy ORM schema for Azure SQL persistence.

Scope, deliberately: this is a PERSISTENCE layer over data that already
has an authoritative shape and meaning defined elsewhere in this
repository --
    - member/ed_visit/care_record columns mirror raw_members.csv /
      raw_ed_visits.csv / raw_care_history.csv exactly (see csv_info.txt
      and backend/main.py's UC07_*_REQUIRED lists) -- no column is
      invented here.
    - safety_context_records mirrors backend/agents/safety_context_csv.py's
      OPTIONAL_SAFETY_COLUMNS -- current-context-only, never a model
      feature (see that module's docstring).
    - analysis_results mirrors backend/agents/contracts.py's
      FinalUC07Decision exactly, field for field -- this table stores an
      ALREADY-COMPUTED decision, it never computes one itself.

Nothing in this module imports, calls, or duplicates any agent/model
logic. It only stores and retrieves.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class UserSession(Base):
    """A server-side session row. The cookie holds only `id` (an opaque
    random token) -- everything else about "who is asking" is looked up
    here, never trusted from the client."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Population(Base):
    """One saved, analyzed dataset. `owner_user_id` is the sole ownership
    boundary every repository query filters on -- never derived from
    anything the client sends."""

    __tablename__ = "populations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    source_members_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_ed_visits_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_care_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_safety_context_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # The index_date the ORIGINAL analysis used -- reload never
    # recomputes, but this is retained as audit/display metadata (see
    # docs requirement: display "captured as of <date>").
    index_date: Mapped[dt.date] = mapped_column(nullable=False)

    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(100), nullable=False)
    synthetic_model: Mapped[bool] = mapped_column(Boolean, nullable=False)

    member_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    members: Mapped[list["PopulationMember"]] = relationship(cascade="all, delete-orphan", passive_deletes=True)
    ed_visits: Mapped[list["PopulationEdVisit"]] = relationship(cascade="all, delete-orphan", passive_deletes=True)
    care_records: Mapped[list["PopulationCareRecord"]] = relationship(cascade="all, delete-orphan", passive_deletes=True)
    safety_context_records: Mapped[list["PopulationSafetyContext"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )
    analysis_results: Mapped[list["AnalysisResult"]] = relationship(cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (Index("ix_populations_owner_created", "owner_user_id", "created_at"),)


class PopulationMember(Base):
    """One row per member_id per population -- mirrors raw_members.csv
    exactly (see UC07_MEMBERS_REQUIRED in backend/main.py)."""

    __tablename__ = "population_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    population_id: Mapped[int] = mapped_column(ForeignKey("populations.id", ondelete="CASCADE"), nullable=False)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False)

    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(20), nullable=False)
    diabetes: Mapped[int] = mapped_column(Integer, nullable=False)
    copd: Mapped[int] = mapped_column(Integer, nullable=False)
    hypertension: Mapped[int] = mapped_column(Integer, nullable=False)
    chf: Mapped[int] = mapped_column(Integer, nullable=False)
    asthma: Mapped[int] = mapped_column(Integer, nullable=False)
    ckd: Mapped[int] = mapped_column(Integer, nullable=False)
    num_chronic_conditions: Mapped[int] = mapped_column(Integer, nullable=False)
    transportation_barrier: Mapped[int] = mapped_column(Integer, nullable=False)
    telehealth_available: Mapped[int] = mapped_column(Integer, nullable=False)
    pcp_distance_miles: Mapped[float] = mapped_column(Float, nullable=False)
    urgent_care_distance_miles: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("population_id", "member_id", name="uq_population_members_member"),
        Index("ix_population_members_population_member", "population_id", "member_id"),
    )


class PopulationEdVisit(Base):
    """Mirrors raw_ed_visits.csv exactly (see UC07_ED_REQUIRED)."""

    __tablename__ = "population_ed_visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    population_id: Mapped[int] = mapped_column(ForeignKey("populations.id", ondelete="CASCADE"), nullable=False)
    visit_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False)
    visit_date: Mapped[dt.date] = mapped_column(nullable=False)
    diagnosis: Mapped[str | None] = mapped_column(String(200), nullable=True)
    triage_level: Mapped[int] = mapped_column(Integer, nullable=False)
    admitted: Mapped[int] = mapped_column(Integer, nullable=False)
    icu: Mapped[int] = mapped_column(Integer, nullable=False)
    major_procedure: Mapped[int] = mapped_column(Integer, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)
    red_flag: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (Index("ix_population_ed_visits_population_member", "population_id", "member_id"),)


class PopulationCareRecord(Base):
    """Mirrors raw_care_history.csv exactly (see UC07_CARE_REQUIRED)."""

    __tablename__ = "population_care_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    population_id: Mapped[int] = mapped_column(ForeignKey("populations.id", ondelete="CASCADE"), nullable=False)
    care_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False)
    visit_date: Mapped[dt.date] = mapped_column(nullable=False)
    care_type: Mapped[str] = mapped_column(String(50), nullable=False)

    __table_args__ = (Index("ix_population_care_records_population_member", "population_id", "member_id"),)


class PopulationSafetyContext(Base):
    """Optional 4th-CSV data (current_safety_context.csv), current-status
    only -- see backend/agents/safety_context_csv.py. A NULL field means
    unknown, never 0/false, matching that module's semantics exactly.
    `captured_at` lets the UI show "Current status captured at <ts>" per
    the requirement that stale current-status never silently masquerades
    as freshly supplied."""

    __tablename__ = "population_safety_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    population_id: Mapped[int] = mapped_column(ForeignKey("populations.id", ondelete="CASCADE"), nullable=False)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False)

    red_flag: Mapped[int | None] = mapped_column(Integer, nullable=True)
    icu: Mapped[int | None] = mapped_column(Integer, nullable=True)
    admitted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    major_procedure: Mapped[int | None] = mapped_column(Integer, nullable=True)
    triage_level: Mapped[int | None] = mapped_column(Integer, nullable=True)

    captured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("population_id", "member_id", name="uq_population_safety_context_member"),)


class AnalysisResult(Base):
    """One row per member per population: the FULL, already-computed
    FinalUC07Decision (backend/agents/contracts.py), stored verbatim.
    Reload/search/pagination/filtering read ONLY this table -- the model
    is never rerun for those operations. List-shaped fields (which don't
    exist as their own queryable dimension) are stored as JSON text;
    everything actually filtered/sorted/searched on is a real column."""

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    population_id: Mapped[int] = mapped_column(ForeignKey("populations.id", ondelete="CASCADE"), nullable=False)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # RiskAssessment
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    tier: Mapped[str] = mapped_column(String(20), nullable=False)
    contributing_factors_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(100), nullable=False)
    synthetic_model: Mapped[bool] = mapped_column(Boolean, nullable=False)
    index_date: Mapped[dt.date] = mapped_column(nullable=False)
    moderate_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    high_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    explanation_factors_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    explanation_method: Mapped[str] = mapped_column(String(50), nullable=False)
    explanation_causal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # FinalNavigationView
    navigation_destination: Mapped[str | None] = mapped_column(String(50), nullable=True)
    navigation_reason_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    navigation_explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # SafetyDecision
    safety_state: Mapped[str] = mapped_column(String(20), nullable=False)
    safety_override: Mapped[bool] = mapped_column(Boolean, nullable=False)
    safety_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    safety_blocked_phrases_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    safety_context_completeness: Mapped[str] = mapped_column(String(20), nullable=False)
    safety_context_source: Mapped[str] = mapped_column(String(30), nullable=False)

    disclaimer: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (
        UniqueConstraint("population_id", "member_id", name="uq_analysis_results_member"),
        Index("ix_analysis_results_population_tier", "population_id", "tier"),
        Index("ix_analysis_results_population_safety", "population_id", "safety_state"),
        Index("ix_analysis_results_population_member", "population_id", "member_id"),
    )
