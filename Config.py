def set_config(args, custom_dataset="haha"):
    available_dataaset = ['Imagenet', 'CIFAR10', 'MNIST']
    if custom_dataset not in available_dataaset:
        ValueError(f'{custom_dataset} is not availabel, please select from {available_dataaset}')


    print(f"Config : {custom_dataset}")

    # Load reduced dataset model instead
    if args.load_different_model:
        training_load_weight = f"rd{int(args.rd * 100)}_c{args.split_class[0]}_ckpt_{args.load_rd_training_checkpoint}_.pt"
        test_load_weight = f"rd{int(args.rd * 100)}_c{args.split_class[0]}_ckpt_{args.load_rd_training_checkpoint}_.pt"
    else:
        training_load_weight = f"{str(args.split_class[0])}_ckpt_{args.load_training_checkpoint}_.pt"
        test_load_weight = f"c{str(args.split_class[0])}_ckpt_{args.load_training_checkpoint}_.pt"
    
    #print(f"training_load_weight : {training_load_weight}")




    model_config = {
        "dataset" : args.dataset,
        "state": args.state,#"train", # or eval
        "epoch": args.epochs,
        "batch_size": args.batch_size,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 128,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": training_load_weight,
        "save_weight_dir": f"{str(args.save_weight_dir)}",#'/kaggle/input/mnist-interference',#'/kaggle/input/cifar10',
        "load_weight_dir": f"{str(args.save_weight_dir)}",#'/kaggle/input/mnist-interference',#'/kaggle/input/cifar10',
        "test_load_weight": test_load_weight,
        "sampled_dir": args.sample_dir + args.dataset +"/",
        "sampledNoisyImgName": f"NoisyNoGuidenceImgs{args.load_training_checkpoint+args.epochs}.png",
        "sampledImgName": f"SampledNoGuidenceImgs{args.load_training_checkpoint+args.epochs}.png",
        "nrow": 8
    }
    
    return model_config 