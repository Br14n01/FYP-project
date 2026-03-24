"""
Random Forest baseline – used via src.ml.train; kept for reference.
"""

from sklearn.ensemble import RandomForestClassifier


def random_forest(n_estimators: int = 100, random_state: int = 42):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
