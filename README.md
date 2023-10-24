AutoML for binary classification.  
Uses H2O AutoML and Hydra for configs.

OmegaConf's Structured Configs are used mainly as a schema for yaml files validation.

Function `process_num_df_to_binaryclass` converts numeric labels in a dataframe to binary classes and filters out features with values below low_num_feature_val_thr in more than `low_num_feature_samples_thr` samples.