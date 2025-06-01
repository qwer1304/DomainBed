python -m domainbed.scripts.train\
       --data_dir=./domainbed/data/MNIST/\
       --algorithm GLSD_SSD\
       --dataset ColoredMNIST\
       --test_env 4\
       --checkpoint_freq 1\
       --save_model_every_checkpoint\
       --steps 5000\
       --hparams '{"glsd_gamma": 15, "batch_size": 128, "resnet_dropout": 0.5, "weight_decay": 1e-3}'