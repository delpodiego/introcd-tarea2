from typing import Any, Literal


def get_model(
    model_name: Literal["naive-bayes", "decision-tree"],
    params: dict[str, Any] | None = None,
):
    if params is None:
        params = {}

    if model_name == "naive-bayes":
        from sklearn.naive_bayes import MultinomialNB
        return MultinomialNB(**params)

    elif model_name == "decision-tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**params)

    else:
        raise ValueError(f"Nombre de modelo inválido: {model_name}")
