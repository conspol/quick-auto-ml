import math
import sys
from pathlib import Path
from typing import Dict, List, Union

import h2o
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
from hydra.core.config_store import ConfigStore
from loguru import logger as lg
from omegaconf import OmegaConf

from quick_auto_ml.conf_schema.structured_configs import (
    AppConfig,
    MergeWithConfig,
)
from quick_auto_ml.covariances import (
    CovarianceMethod,
    EmpCov,
    MinCov,
    NumpyCov,
    ShrunkCov,
)
from quick_auto_ml.data_proc import (
    get_num_features_data,
    load_dataframe,
    prepare_data,
    prepare_test_train,
    process_num_df_to_binaryclass,
)
from quick_auto_ml.defines import CLASS_LABEL
from quick_auto_ml.plots import show_feature_matrix
from quick_auto_ml.utils import process_random_seeds, process_same_file_cfgs

config_store = ConfigStore.instance()
config_store.store(name='base_config', node=AppConfig)


@hydra.main(
    config_path='quick_auto_ml/conf',
    config_name="_private_config",
    version_base=None,
)
def main(
    cfg: AppConfig,
) -> None:
    lg.debug(f"\n{OmegaConf.to_yaml(cfg)}")

    lg.configure(handlers=[{
        'sink': sys.stdout,
        'level': cfg.log_level.name,
    }])

    # ============= Prepare configs =============

    if cfg.change_nested_random_seeds:
        process_random_seeds(cfg)

    cfg_ds = cfg.data
    process_same_file_cfgs(cfg_ds)

    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    # ============= END Prepare configs =============


    input_file_path = Path(cfg_ds.input_file)

    if (not cfg_ds.input_file_type.lower() == 'excel' 
        or not input_file_path.suffix == '.xlsx'):
        raise NotImplementedError(
            "Only xlsx input files are supported at this time. "
        )

    data = load_dataframe(cfg_file=cfg_ds)

    lg.debug("Loaded dataframe:")
    lg.debug(f"\n{data}")

    data = prepare_data(data=data, cfg_ds=cfg_ds)

    lg.debug("Processed dataframe:")
    lg.debug(f"\n{data}")

    # TODO: test data preparation (split, from file, ...)

    train_df, test_df = prepare_test_train(
        data=data,
        cfg_ds=cfg_ds,
    )

    h2o.init(
        nthreads=10,
        max_mem_size='12G',
    )

    hf = h2o.H2OFrame(train_df)
    test_hf = h2o.H2OFrame(test_df)
    exclude_cols = {cfg_ds.label_column, CLASS_LABEL, *cfg.train.exclude_cols}
    feature_cols = [col_ for col_ in train_df.columns if col_ not in exclude_cols]

    nfolds = hf.nrows
    aml = H2OAutoML(
        max_models=50,
        max_runtime_secs=60,
        seed=cfg.random_seed,
        nfolds=nfolds,
    )

    hf['Diagnosis'] = hf['Diagnosis'].asfactor()

    aml.train(
        x=feature_cols,
        y=CLASS_LABEL,
        # y=cfg_ds.label_column,
        training_frame=hf,
    )
    lb = aml.leaderboard

    print(lb.as_data_frame())


    for m_ in lb.as_data_frame().values:
        mdl_ = h2o.get_model(m_[0])
        print(mdl_.mse())

    model_id = list(lb['model_id'].as_data_frame().iloc[:, 0])[6]
    mdl = h2o.get_model(model_id)
    mdl.shap_summary_plot(hf)
    pred_res = mdl.predict(test_hf)
    pred_res.as_data_frame()


#     # ppp = mdl.predict(hf).head(32).as_data_frame()
#     # print(data.reset_index().join(ppp).iloc[:,-5:])

#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(10, 8))
#     for model_id in lb['model_id'].as_data_frame().values:
#         model = h2o.get_model(model_id[0])
#         perf = model.model_performance(hf)
#         fpr = perf.fprs
#         tpr = perf.tprs
#         plt.plot(fpr, tpr, label=model_id[0])

#     # Plot random guess line
#     plt.plot([0, 1], [0, 1], linestyle='--', color='black')

    from h2o.estimators import (
        H2OGradientBoostingEstimator,
        H2ORandomForestEstimator,
        H2OSupportVectorMachineEstimator,
        H2OXGBoostEstimator,
    )
    from h2o.tree import H2OTree

    gb = H2OGradientBoostingEstimator(
        ntrees=150,
        nfolds=6,
        max_depth=30,
        keep_cross_validation_predictions=True,
    )
    gb.train(x=feature_cols,
            #  y=cfg_ds.label_column,
             y=CLASS_LABEL,
             training_frame=hf)
    print(gb.cross_validation_metrics_summary().as_data_frame())
    gb.shap_summary_plot(hf)
    gb.auc()

    xgb = H2OXGBoostEstimator(
        nfolds=6,
        min_rows=1,
        keep_cross_validation_predictions=True,
    )
    xgb.train(x=feature_cols,
             y=cfg_ds.label_column,
            #  y=CLASS_LABEL,
             training_frame=hf)
    print(xgb.cross_validation_metrics_summary().as_data_frame())
    xgb.shap_summary_plot(hf)
    xgb.auc()

    rf = H2ORandomForestEstimator(
        # ntrees=1,
        nfolds=8,
        sample_rate=1,
        mtries=15,
        max_depth=200,
        keep_cross_validation_predictions=True,
    )
    rf.train(x=feature_cols,
            #  y=cfg_ds.label_column,
             y=CLASS_LABEL,
             training_frame=hf)
    print(rf.cross_validation_metrics_summary().as_data_frame())
    rf.shap_summary_plot(hf)
    rf.auc()


    import seaborn as sns
    from sklearn import svm
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    exclude_cols_svm = exclude_cols.union({'Diagnosis'})
    feature_cols = [col_ for col_ in data.columns if col_ not in exclude_cols_svm]
    feat_data_scaled = StandardScaler().fit_transform(data[feature_cols])

    num_data = get_num_features_data(
        data=data,
        label_column=CLASS_LABEL,
        preserve_label=False,
    )
    scaled_data = pd.DataFrame(feat_data_scaled, columns=feature_cols, index=data.index)


    import dash
    from dash import dcc, html


    num_data.boxplot()
    plt.xticks(rotation=90)


    sns.boxplot(scaled_data)
    sns.stripplot(scaled_data, dodge=True, jitter=True, color='.3')
    plt.xticks(rotation=90)
    plt.show()


    methods: List[Dict[str, Union[str, CovarianceMethod]]] = [
        {"name": "Shrunk Covariance", "instance": ShrunkCov(feat_data_scaled)},
        {"name": "Minimum Covariance Determinant", "instance": MinCov(feat_data_scaled)},
        {"name": "Empirical Covariance", "instance": EmpCov(feat_data_scaled)},
        # Should match the empirical covariance
        {"name": "NumPy Covariance", "instance": NumpyCov(feat_data_scaled)}
    ]

    n_methods = len(methods)
    n_cols = 2
    n_rows = math.ceil(n_methods / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows*5))


    for idx, method in enumerate(methods):
        ax = axs[idx // n_cols, idx % n_cols] if n_rows > 1 else axs[idx % n_cols]
        covariance_matrix = method["instance"].get_covariance()
        show_feature_matrix(covariance_matrix, feature_cols, ax, title=method["name"])

    # Remove any unused subplots
    if n_methods < n_rows * n_cols:
        for j in range(n_methods, n_rows * n_cols):
            fig.delaxes(axs.flatten()[j])

    plt.tight_layout()

    pca = PCA(n_components=10)
    pca.fit(feat_data_scaled)

    train, test = train_test_split(
        data, test_size=0.2,
        # random_state=cfg.random_seed,
        stratify=data[CLASS_LABEL],
    )

    exclude_cols_svm = exclude_cols.union({'Diagnosis'})
    feature_cols = [col_ for col_ in data.columns if col_ not in exclude_cols_svm]
    X_train = train[feature_cols]
    y_train = train[CLASS_LABEL]
    X_test = test[feature_cols]
    y_test = test[CLASS_LABEL]

    scaler = StandardScaler()

    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # X_train_scaled = X_train**2
    # X_test_scaled = X_test**2

    X_train_scaled = scaler.fit_transform(2**X_train)
    X_test_scaled = scaler.transform(2**X_test)

    # X_train_scaled = X_train
    # X_test_scaled = X_test

    clf = svm.NuSVC(
        gamma='auto',
        kernel='rbf',
        # random_state=cfg.random_seed
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(classification_report(y_test, y_pred))



    xx, yy = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
    # plot the decision function for each datapoint on the grid

    feature_1 = 'BCL2-BIMBH3'
    feature_2 = 'BCL2-BAD'

    fixed_features = {feature: -3.4
                      for feature in feature_cols
                      if feature not in [feature_1, feature_2]}

    Z = np.array([clf.decision_function(
        np.array([[x, y] + list(fixed_features.values())]))
        for x, y in zip(np.ravel(xx), np.ravel(yy))])

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, cmap=plt.cm.bwr)
    plt.scatter(data[feature_1], data[feature_2], edgecolors='k')  # Assuming 'target' is the name of your target variable
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.show()

    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)


    # svm = H2OSupportVectorMachineEstimator()
    # svm.train(x=feature_cols, y=CLASS_LABEL, training_frame=hf)
    # print(svm.cross_validation_metrics_summary().as_data_frame())



    lg.info("fin.")


if __name__ == '__main__':
    main()