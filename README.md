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

