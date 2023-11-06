# Intro
AutoML for binary classification.  
Uses H2O AutoML and Hydra for configs.

OmegaConf's Structured Configs are used mainly as a schema for yaml files validation.


# Docs (sort of)
## Random Seed Behavior
If `random_seed` is set in the root config, all seeds in the nested configurations
are set to the root's value (by the `process_random_seeds` utility function),
except for those explicitly set to `-1`. The ones marked with `-1` will be set 
to `None`, thus preserving the random behavior. 

Enforcing root config's `random_seed` on children configurations can be turned
off by setting `change_nested_random_seeds: false` in .yaml config file.


## Same File Configuration
When children data file configurations are present (such as `merge_data` or `test_data.file`), user can set them to inherit the main data file's properties.
If `input_file` is set to `'same'`, the utility function (`process_same_file_cfgs`)
copies the parent's file path and type to the child configuration.

When the `input_file_sheet_name` is `'same'`
or not provided, it adopts the sheet name from the parent configuration.
In this case `skiprows` and `skipfooter` parameters are also synchronized if
not provided.


## Train-Test Split Configuration
The configuration of test data splitting is handled by specifying configuration
under `test_data`. Three options are available: `split`, `file`, or `index_file`.
Only one option can be set at a time, otherwise an error will be raised. 

### Split Behavior
When `test_data.split` is used, the dataset is split into training and testing
sets based on the specified `test_size`, `random_seed`, and optional `stratify_by`
column.

### File-Based Test Data
Setting `test_data.file` allows for loading a separate test dataset from a specified file. This method is not yet fully implemented and raises a `NotImplementedError`.

### Index-Based Split
The `test_data.index_file` option enables splitting based on indices provided in an some file. The main dataframe is split into training and test dataframes where the test set comprises rows with indices matching those in the specified index file.


## Reference manual
Function `process_num_df_to_binaryclass` converts numeric labels in a dataframe to binary classes and filters out features with values below `low_num_feature_val_thr` in more than `low_num_feature_samples_thr` samples.
