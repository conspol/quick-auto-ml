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
