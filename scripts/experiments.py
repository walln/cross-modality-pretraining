from cross_modality_learning.experiment import run_experiment

RETRAIN_LR = 1e-7
PRETRAIN_LR = 1e-3
NO_PRETRAIN_LR = 1e-3
PATCH_SIZE = 4
BATCH_SIZE = 8

CIFAR = "cifar10"
MNIST = "mnist"
IMDB = "imdb"

GPT = "gpt2"
VIT = "vit"

PRETRAINED = "pretrained"
RETRAINED = "retrained"
NO_PRETRAIN = "no_pretrain"

def create_config(type, task, model):
    if type == NO_PRETRAIN:
        params = dict(
            task=task,
            model_name=model,
            pretrained=False,
            learning_rate=NO_PRETRAIN_LR,
            batch_size=BATCH_SIZE,
            patch_size=PATCH_SIZE,
            freeze_trans=False,
            freeze_in=False,
            freeze_pos=False,
            freeze_ln=False,
            freeze_attn=False,
            freeze_ff=False,
            freeze_out=False
        )
        return params
    elif type == PRETRAINED:
        params = dict(
            task=task,
            model_name=model,
            pretrained=True,
            learning_rate=NO_PRETRAIN_LR,
            batch_size=BATCH_SIZE,
            patch_size=PATCH_SIZE,
            freeze_trans=True, 
            freeze_in=False,
            freeze_pos=False,
            freeze_ln=False,
            freeze_attn=True,
            freeze_ff=True,
            freeze_out=False
        )
        return params  
    elif type == RETRAINED:
        params = dict(
            task=task,
            model_name=model,
            pretrained=True,
            learning_rate=NO_PRETRAIN_LR,
            batch_size=BATCH_SIZE,
            patch_size=PATCH_SIZE,
            freeze_trans=False, 
            freeze_in=False,
            freeze_pos=False,
            freeze_ln=False,
            freeze_attn=False,
            freeze_ff=False,
            freeze_out=False
        )
        return params


def gpt2_pretrain_cifar():
    experiment_name = 'gpt2_pretrained_cifar'
    experiment_params = create_config(PRETRAINED, CIFAR, GPT)
    run_experiment(experiment_name, experiment_params)

def vit_pretrain_cifar():
    experiment_name = 'vit_pretrained_cifar'
    experiment_params = create_config(PRETRAINED, CIFAR, VIT)
    run_experiment(experiment_name, experiment_params)

def gpt2_full_train_cifar():
    experiment_name = 'gpt2_no_pretrain_cifar'
    experiment_params = create_config(NO_PRETRAIN, CIFAR, GPT)
    run_experiment(experiment_name, experiment_params)

def vit_full_train_cifar():
    experiment_name = 'gpt2_no_pretrained_cifar'
    experiment_params = create_config(NO_PRETRAIN, CIFAR, VIT)
    run_experiment(experiment_name, experiment_params)

def gpt2_retrain_cifar():
    experiment_name = 'gpt2_retrained_cifar'
    experiment_params = create_config(RETRAINED, CIFAR, GPT)
    run_experiment(experiment_name, experiment_params)

def vit_retrain_cifar():
    experiment_name = 'vit_retrained_cifar'
    experiment_params = create_config(RETRAINED, CIFAR, VIT)
    run_experiment(experiment_name, experiment_params)

def gpt2_pretrain_mnist():
    experiment_name = 'gpt2_pretrained_mnist'
    experiment_params = create_config(PRETRAINED, MNIST, GPT)
    run_experiment(experiment_name, experiment_params)

def vit_pretrain_mnist():
    experiment_name = 'vit_pretrained_mnist'
    experiment_params = create_config(PRETRAINED, MNIST, VIT)
    run_experiment(experiment_name, experiment_params)

def gpt2_full_train_mnist():
    experiment_name = 'gpt2_no_pretrain_mnist'
    experiment_params = create_config(NO_PRETRAIN, MNIST, GPT)
    run_experiment(experiment_name, experiment_params)

def vit_full_train_mnist():
    experiment_name = 'vit_no_pretrain_mnist'
    experiment_params = create_config(NO_PRETRAIN, MNIST, VIT)
    run_experiment(experiment_name, experiment_params)

def gpt2_retrain_mnist():
    experiment_name = 'gpt2_retrained_mnist'
    experiment_params = create_config(RETRAINED, MNIST, GPT)
    run_experiment(experiment_name, experiment_params)   

def vit_retrain_mnist():
    experiment_name = 'vit_retrained_mnist'
    experiment_params = create_config(RETRAINED, MNIST, VIT)
    run_experiment(experiment_name, experiment_params)

def gpt2_pretrain_imdb():
    experiment_name = 'gpt2_pretrained_imdb'
    experiment_params = create_config(PRETRAINED, IMDB, GPT)
    run_experiment(experiment_name, experiment_params)

def vit_pretrain_imdb():
    experiment_name = 'vit_pretrained_imdb'
    experiment_params = create_config(PRETRAINED, IMDB, VIT)
    run_experiment(experiment_name, experiment_params)

def gpt2_full_train_imdb():
    experiment_name = 'gpt2_no_pretrain_imdb'
    experiment_params = create_config(NO_PRETRAIN, IMDB, GPT)
    run_experiment(experiment_name, experiment_params)

def vit_full_train_imdb():
    experiment_name = 'vit_no_pretrain_imdb'
    experiment_params = create_config(NO_PRETRAIN, IMDB, VIT)
    run_experiment(experiment_name, experiment_params)

def gpt2_retrain_imdb():
    experiment_name = 'gpt2_retrained_imdb'
    experiment_params = create_config(RETRAINED, IMDB, GPT)
    run_experiment(experiment_name, experiment_params)   

def vit_retrain_imdb():
    experiment_name = 'vit_retrained_imdb'
    experiment_params = create_config(RETRAINED, IMDB, VIT)
    run_experiment(experiment_name, experiment_params)


# if __name__ == '__main__':
#     gpt2_retrain_cifar()
#     vit_retrain_cifar()

