python -m domainbed.scripts.train^
       --data_dir=./domainbed/data/MNIST/^
       --algorithm GLSD_FSD^
       --dataset ColoredMNIST^
       --save_model_every_checkpoint^
       --checkpoint_freq 1^
       --test_env 4