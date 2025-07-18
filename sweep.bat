python -m domainbed.scripts.sweep launch^
       --data_dir=./domainbed/data/MNIST/^
       --output_dir=./domainbed/sweep/^
       --command_launcher local^
       --algorithms GLSD_SSD^
       --datasets ColoredMNIST^
       --n_hparams 3^
       --specific_test_envs 3 4^
       --specific_test_envs 0 4^
       --n_trials 1^
       --skip_confirmation^
       --checkpoint_freq 100^
       --save_model_every_checkpoint^
       --colwidth 10