import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--drop", default=None, help="column to remove from the features")
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)
    target = data["Churn"]
    features = data.drop(columns="Churn")

    
    if args.drop:
        if args.drop not in features.columns:
            raise SystemExit(f"error: column '{args.drop}' not found")
        features = features.drop(columns=args.drop)


    text_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    numeric_columns = features.columns.difference(text_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                text_columns,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    print(f"accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
