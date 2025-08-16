from promptlearn.benchmark import run_benchmark
from pathlib import Path
import os

from promptlearn.benchmark import run_benchmark, make_sklearn, make_promptlearn_variant

BASE = Path(os.environ.get("CCBENCH_TASKS_DIR", "../ccbench-tasks/tasks"))
TASKS = [
    BASE/"animals/is_mammal",
    BASE/"physics/fall_time",
    BASE/"bool/xor",
    BASE/"equations/linear_y_2x_plus_3",
    BASE/"equations/noisy_linear_y_2x_plus_3",
    BASE/"equations/nonlinear_two_vars",
    BASE/"flags/has_blue_in_flag",
    BASE/"flags/zero_training_rows",
    BASE/"riddles/animal_money",
]

models = [
    # --- classification ---
    make_sklearn("LogReg", "sklearn.linear_model.LogisticRegression", max_iter=1000),
    make_sklearn("RF", "sklearn.ensemble.RandomForestClassifier", n_estimators=300, random_state=0),
    make_promptlearn_variant("classifier", llm_name="gpt-4o", name="PromptLearnClf[gpt-4o]"),
    make_promptlearn_variant("classifier", llm_name="gpt-5",  name="PromptLearnClf[gpt-5]"),

    # --- regression ---
    make_sklearn("LinReg", "sklearn.linear_model.LinearRegression"),
    make_sklearn("RFReg", "sklearn.ensemble.RandomForestRegressor", n_estimators=300, random_state=0),
    make_promptlearn_variant("regressor", llm_name="gpt-4o", name="PromptLearnReg[gpt-4o]"),
    make_promptlearn_variant("regressor", llm_name="gpt-5",  name="PromptLearnReg[gpt-5]"),
]

cls_df, reg_df = run_benchmark(TASKS, models=models, out_dir="runs", resume=True, return_kind="split")

print("\n=== Classification ===")
print(cls_df.round(4))
print("\n=== Regression ===")
print(reg_df.round(4))
