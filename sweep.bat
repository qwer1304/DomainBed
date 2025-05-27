python -m domainbed.scripts.sweep launch^
       --data_dir=./domainbed/data/MNIST/^
       --output_dir=./domainbed/output/^
       --command_launcher local^
       --algorithms GLSD^
       --datasets ColoredMNIST^
       --n_hparams 3^
       --single_test_envs^
       --n_trials 1