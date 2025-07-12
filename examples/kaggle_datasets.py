import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from promptlearn.classifier import PromptClassifier

# === Load and preprocess data ===
# Download from https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/code
df = pd.read_csv("examples/external_data/personality_datasert.csv")

# --- General code for evaluating models below ---

# Automatically pick the last column as the target
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode for classification scoring
y = LabelEncoder().fit_transform(y)

# Stratified split for fair validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Define model/grid combinations ===
model_names = ["gpt-4o", "o3-mini", "o4-mini", "gpt-4.1-mini", "gpt-4.1"]

results = []
print(
    "⚠️ Warning: This will make multiple LLM calls per model/grid! Monitor API cost.\n"
)

for name in model_names:
    clf = PromptClassifier(model=name)
    grid = {"model": [name], "max_train_rows": [100, 500, 1000, 2000, X_train.shape[0]]}
    gs = GridSearchCV(
        clf,
        param_grid=grid,
        cv=2,  # Save some money by using a smaller CV
        scoring="f1_weighted",
        n_jobs=1,  # Set to 1 to avoid accidental LLM parallel calls unless you know what you are doing!
        verbose=1,
        error_score="raise",
    )
    gs.fit(X_train, y_train)
    # Evaluate on held-out set
    y_pred = gs.best_estimator_.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred, average="weighted")
    results.append(
        {
            "model": name,
            "best_cv_f1": gs.best_score_,
            "best_params": gs.best_params_,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
        }
    )
    print(
        f"\n{name}: Best params {gs.best_params_} | CV F1 {gs.best_score_:.3f} | Val Accuracy {val_acc:.3f} | Val F1 {val_f1:.3f}"
    )

# === Display summary results ===
df_results = pd.DataFrame(results).sort_values("val_f1", ascending=False)
print("\nModel Performance (Sorted by Validation F1):\n")
print(df_results.to_string(index=False))

# Model Performance (Sorted by Validation F1):
#
#        model  best_cv_f1                                       best_params  val_accuracy   val_f1
#       gpt-4o    0.908088       {'max_train_rows': 2000, 'model': 'gpt-4o'}      0.917241 0.917248
# gpt-4.1-mini    0.938375 {'max_train_rows': 2000, 'model': 'gpt-4.1-mini'}      0.908621 0.908638
#      o4-mini    0.938375      {'max_train_rows': 1000, 'model': 'o4-mini'}      0.893103 0.893103
#      gpt-4.1    0.935358      {'max_train_rows': 2320, 'model': 'gpt-4.1'}      0.884483 0.884447
#      o3-mini    0.835065      {'max_train_rows': 2000, 'model': 'o3-mini'}      0.513793 0.348771
