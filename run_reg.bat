python -m domainbed.scripts.train^
       --data_dir=./domainbed/data/MNIST/^
       --algorithm GLSD_SSD^
       --dataset ColoredMNIST^
       --save_model_every_checkpoint^
       --checkpoint_freq 1^
       --test_env 4^
       --steps 2500^
       --hparams "{\"glsd_as_regularizer\": true, \"glsd_dominate_all_domains\": true}"
