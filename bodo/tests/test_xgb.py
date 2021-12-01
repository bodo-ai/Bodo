import pytest
from sklearn.datasets import make_classification

from bodo.tests.utils import check_func


@pytest.mark.skip("CI cannot test XGBoost yet")
def test_xgb_classifier_predict_proba():
    import xgboost as xgb

    xgb.rabit.init()
    splitN = 500
    n_samples = 1000
    n_features = 50
    X_train = None
    y_train = None
    X_test = None
    # Create exact same dataset on all ranks
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=3,
        n_clusters_per_class=2,
        n_informative=3,
        random_state=20,
    )
    X_train = X[:splitN]
    y_train = y[:splitN]
    X_test = X[splitN:]

    # Create exact same model on all ranks using xgboost directly.
    # That way, we can test predict_proba and predict_log_proba implementation
    # independent of the model.
    # Note that XGBoost has native MPI support, so this is not exactly correct
    # since it's expecting distributed data and we're giving it replicated data.
    # But eventual model should be about the same since the data is just duplicated.
    clf = xgb.XGBClassifier(
        booster="gbtree",
        objective="binary:logistic",
        random_state=0,
        tree_method="hist",
    )

    clf.fit(X_train, y_train)

    def impl_predict_proba(clf, X_test):
        y_pred_proba = clf.predict_proba(X_test)
        return y_pred_proba

    check_func(impl_predict_proba, (clf, X_test))


@pytest.mark.skip("CI cannot test XGBoost yet")
def test_xgb_feature_importances_():
    import xgboost as xgb

    xgb.rabit.init()
    splitN = 500
    n_samples = 1000
    n_features = 50
    X_train = None
    y_train = None
    X_test = None
    # Create exact same dataset on all ranks
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=3,
        n_clusters_per_class=2,
        n_informative=3,
        random_state=20,
    )
    X_train = X[:splitN]
    y_train = y[:splitN]

    # See note in test_xgb_classifier_predict_proba
    clf = xgb.XGBClassifier(
        booster="gbtree",
        objective="binary:logistic",
        random_state=0,
        tree_method="hist",
    )

    clf.fit(X_train, y_train)

    def impl(clf):
        return clf.feature_importances_

    check_func(
        impl,
        (clf,),
        is_out_distributed=False,
    )
