import type { MemberFiltersState } from "../tableState";
import { clearFilterField, DEFAULT_FILTERS, describeActiveFilters, isFiltersActive } from "../tableState";
import "./MemberFilters.css";

const PROBABILITY_PRESETS: { label: string; min: string; max: string }[] = [
  { label: "<10%", min: "", max: "10" },
  { label: "10–20%", min: "10", max: "20" },
  { label: "20–30%", min: "20", max: "30" },
  { label: ">30%", min: "30", max: "" },
];

interface MemberFiltersProps {
  filters: MemberFiltersState;
  onChange: (next: MemberFiltersState) => void;
  totalCount: number;
  filteredCount: number;
}

export function MemberFilters({ filters, onChange, totalCount, filteredCount }: MemberFiltersProps) {
  const active = isFiltersActive(filters);
  const chips = describeActiveFilters(filters);

  const set = <K extends keyof MemberFiltersState>(key: K, value: MemberFiltersState[K]) =>
    onChange({ ...filters, [key]: value });

  const applyPreset = (preset: { min: string; max: string }) => {
    onChange({ ...filters, probMin: preset.min, probMax: preset.max });
  };

  return (
    <section className="member-filters" aria-label="Member filters">
      <div className="member-filters__heading">
        <h2 className="member-filters__title">Members</h2>
        <span className="member-filters__count">
          {active ? `${filteredCount} of ${totalCount} members` : `${totalCount} members`}
        </span>
      </div>

      <div className="member-filters__row">
        <label className="member-filters__field member-filters__field--search">
          <span className="sr-only">Search member ID</span>
          <input
            type="text"
            placeholder="Search member ID…"
            value={filters.search}
            onChange={(e) => set("search", e.target.value)}
          />
        </label>

        <label className="member-filters__field">
          <span className="sr-only">Risk tier</span>
          <select value={filters.tier} onChange={(e) => set("tier", e.target.value as MemberFiltersState["tier"])}>
            <option value="ALL">Risk: All</option>
            <option value="LOW">Low</option>
            <option value="MODERATE">Moderate</option>
            <option value="HIGH">High</option>
          </select>
        </label>

        <label className="member-filters__field">
          <span className="sr-only">Navigation</span>
          <select
            value={filters.navigation}
            onChange={(e) => set("navigation", e.target.value as MemberFiltersState["navigation"])}
          >
            <option value="ALL">Navigation: All</option>
            <option value="NO_PROACTIVE_NAVIGATION">No proactive navigation</option>
            <option value="PRIMARY_CARE">Primary Care</option>
            <option value="URGENT_CARE">Urgent Care</option>
            <option value="TELEHEALTH">Telehealth</option>
            <option value="CARE_MANAGEMENT">Care Management</option>
          </select>
        </label>

        <label className="member-filters__field">
          <span className="sr-only">Safety state</span>
          <select value={filters.safety} onChange={(e) => set("safety", e.target.value as MemberFiltersState["safety"])}>
            <option value="ALL">Safety: All</option>
            <option value="CLEAR">Clear</option>
            <option value="CAUTION">Caution</option>
            <option value="OVERRIDE">Override</option>
          </select>
        </label>

        <button
          type="button"
          className="member-filters__clear"
          onClick={() => onChange(DEFAULT_FILTERS)}
          disabled={!active}
        >
          Clear all filters
        </button>
      </div>

      <div className="member-filters__row member-filters__row--probability">
        <span className="member-filters__prob-label">Probability</span>
        <label className="member-filters__prob-input">
          <span className="sr-only">Minimum probability percent</span>
          <input
            type="number"
            min={0}
            max={100}
            placeholder="Min %"
            value={filters.probMin}
            onChange={(e) => set("probMin", e.target.value)}
          />
        </label>
        <span className="member-filters__prob-sep">–</span>
        <label className="member-filters__prob-input">
          <span className="sr-only">Maximum probability percent</span>
          <input
            type="number"
            min={0}
            max={100}
            placeholder="Max %"
            value={filters.probMax}
            onChange={(e) => set("probMax", e.target.value)}
          />
        </label>
        <div className="member-filters__presets">
          {PROBABILITY_PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              className="member-filters__preset"
              onClick={() => applyPreset(preset)}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {chips.length > 0 && (
        <div className="member-filters__chips" aria-label="Active filters">
          <span className="member-filters__chips-label">Active filters:</span>
          {chips.map((chip) => (
            <button
              key={chip.id}
              type="button"
              className="member-filters__chip"
              onClick={() => onChange(clearFilterField(filters, chip.id))}
              aria-label={`Remove filter ${chip.label}`}
            >
              {chip.label} <span aria-hidden="true">×</span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
