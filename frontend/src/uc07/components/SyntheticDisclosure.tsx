import "./SyntheticDisclosure.css";

/** Concise, always-visible synthetic-data disclosure -- required
 * wherever a decision or model identity is shown, per
 * docs/07_DISPARITY_INPUT_SAFETY_HARDENING.md's synthetic-disclosure
 * requirement. Never hidden behind a details toggle. */
export function SyntheticDisclosure({ modelVersion }: { modelVersion?: string }) {
  return (
    <p className="synthetic-disclosure">
      <span className="synthetic-disclosure__badge">Demo</span>
      Demo model trained on synthetic data; not clinically validated.
      {modelVersion && <span className="synthetic-disclosure__version"> ({modelVersion})</span>}
    </p>
  );
}
