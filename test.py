from scripts.experiments import gpt2_full_train_cifar, gpt2_pretrain_cifar, gpt2_retrain_cifar, vit_pretrain_cifar, vit_full_train_cifar, vit_retrain_cifar

if __name__ == "__main__":
    gpt2_pretrain_cifar()
    gpt2_full_train_cifar()
    gpt2_retrain_cifar()
    
    vit_pretrain_cifar()
    vit_full_train_cifar()
    vit_retrain_cifar()
    