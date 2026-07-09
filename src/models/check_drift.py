"""
Data drift gate for CI/CD.
Fails the pipeline if too much of the incoming data has drifted
from the reference training data.
"""
import sys
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

DRIFT_SHARE_THRESHOLD = 0.30  # fail if more than 30% of columns have drifted

reference = pd.read_csv("data/processed/telco_churn_cleaned.csv")

# In real production this would be genuinely new incoming data.
# For now, this demonstrates the gate with a simulated batch.
current = reference.sample(n=500, random_state=1).copy()
current["tenure"] = (current["tenure"] * 0.2).astype(int)
current["MonthlyCharges"] = current["MonthlyCharges"] * 1.4
current["Contract"] = "Month-to-month"

report = Report([DataDriftPreset()])
my_eval = report.run(current_data=current, reference_data=reference)
result = my_eval.dict()

drift_metric = next(m for m in result["metrics"] if m["metric_name"].startswith("DriftedColumnsCount"))
share = drift_metric["value"]["share"]
count = drift_metric["value"]["count"]

my_eval.save_html("drift_report.html")

print(f"Drifted columns: {count} ({share:.1%})")

if share > DRIFT_SHARE_THRESHOLD:
    print(f"FAILED: drift share {share:.1%} exceeds threshold {DRIFT_SHARE_THRESHOLD:.0%}")
    sys.exit(1)
else:
    print(f"PASSED: drift share {share:.1%} within threshold {DRIFT_SHARE_THRESHOLD:.0%}")
    sys.exit(0)
