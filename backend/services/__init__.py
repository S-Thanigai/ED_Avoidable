"""
backend/services/
------------------
Operational communication/reporting services for UC07: PDF report
rendering (report_service.py) and outbound email (email_service.py).

Both modules are deliberately downstream-only consumers of an
ALREADY-COMPUTED decision (RiskAssessment/FinalNavigationView/
SafetyDecision/MemberExplanation, or the frontend's JSON echo of them).
Neither module imports risk_detection.py, care_navigation.py,
safety_policy.decide(), model_explainability.py's SHAP computation, or
orchestrator.py -- they cannot produce, change, or influence a risk
probability, risk tier, navigation destination, or safety state. See
docs/09_MEMBER_COMMUNICATION_REPORTING.md.
"""
