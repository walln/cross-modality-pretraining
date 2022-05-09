import numpy as np
import torch
import wandb
import random
from .model import Model
from .trainer import Trainer
from sklearn.metrics import accuracy_score


def experiment(
        exp_name,
        exp_args,
        **kwargs
):
    assert 'batch_size' in kwargs
    assert kwargs['batch_size'] <= exp_args['gpu_batch_size'] or \
           kwargs['batch_size'] % exp_args['gpu_batch_size'] == 0

    task = kwargs['task']
    batch_size = kwargs['batch_size']
    patch_size = kwargs['patch_size']
    device = exp_args['device']

    max_length = exp_args['max_length'] if 'max_length' in exp_args else 120

    is_binary = False
    return_last_only = True

    if task == 'mnist':
        from .datasets.mnist import MNISTDataset
        dataset = MNISTDataset(batch_size=batch_size, patch_size=patch_size, device=device)
        input_dim, output_dim = patch_size ** 2, 10
        use_embeddings = False
        experiment_type = 'classification'
    elif task == 'cifar10':
        from .datasets.cifar10 import CIFAR10Dataset
        dataset = CIFAR10Dataset(batch_size=batch_size, patch_size=patch_size, device=device)
        input_dim, output_dim = 3 * patch_size**2, 10
        use_embeddings = False
        experiment_type = 'classification'
    elif task == 'imdb':
        from .datasets.imdb import IMDBDataset
        dataset = IMDBDataset(batch_size=batch_size, max_length=max_length, device=device)
        input_dim, output_dim = max_length, 1
        use_embeddings = False
        experiment_type = 'binary'
        is_binary = True
    else: raise NotImplementedError('dataset not implemented')

    if experiment_type == 'classification':
        ce_loss = torch.nn.CrossEntropyLoss()
        def loss_fn(out, y, x=None):
            out = out[:, 0]
            return ce_loss(out, y)
        def accuracy_fn(preds, true, x=None):
            preds = preds[:, 0].argmax(-1)
            return (preds == true).mean()
    elif experiment_type == "binary":
        ce_loss = torch.nn.BCELoss()
        def loss_fn(out, y, x=None):
            out = torch.squeeze(out[:, 0])
            return ce_loss(out, y)
        def accuracy_fn(preds, true, x=None):
            preds = np.round(preds[:, 0])
            return accuracy_score(preds, true)
    else: raise NotImplementedError('experiment_type not recognized')

    model = Model(
        input_dim=input_dim,
        output_dim=output_dim,
        model_name=kwargs.get('model_name', 'gpt2'),
        pretrained=kwargs.get('pretrained', True),
        return_last_only=return_last_only,
        use_embeddings_for_in=use_embeddings,
        freeze_trans=kwargs.get('freeze_trans', True),
        freeze_in=kwargs.get('freeze_in', False),
        freeze_pos=kwargs.get('freeze_pos', False),
        freeze_ln=kwargs.get('freeze_ln', False),
        freeze_attn=kwargs.get('freeze_attn', True),
        freeze_ff=kwargs.get('freeze_ff', True),
        freeze_out=kwargs.get('freeze_out', False),
        is_binary=is_binary
    )
    model.to(device)
    gpu_batch_size = exp_args['gpu_batch_size']
    trainer = Trainer(
        model,
        dataset,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        steps_per_epoch=exp_args['steps_per_iter'],
        test_steps_per_epoch=exp_args['test_steps_per_iter'],
        learning_rate=kwargs['learning_rate'],
        batch_size=gpu_batch_size if batch_size > gpu_batch_size else batch_size,
        eval_batch_size=batch_size,
        grad_accumulate=batch_size // gpu_batch_size if batch_size > gpu_batch_size else 1,
    )

    log_to_wandb = exp_args['log_to_wandb']
    save_models = exp_args['save_models']
    wandb_project = exp_args['wandb_project']

    short_name = str(random.randint(int(1e5), int(1e6) - 1))
    run_name = f'{exp_name}-{task}-{short_name}'

    run = None

    if log_to_wandb:
        config = dict(
            short_name=short_name,
            run_name=run_name,
            **exp_args,
            **kwargs,
        )
        run = wandb.init(
            name=f'{exp_name}-{short_name}',
            group=f'{exp_name}-{task}',
            project=wandb_project,
            config=config,
            reinit=True
        )
        wandb.watch(model)

    print(f'Running for {exp_args["num_iters"]} iters')
    for t in range(exp_args['num_iters']):
        trainer.train_epoch()

        print('=' * 57)
        print(f'| Iteration {" " * 15} | {t+1:25} |')
        for k, v in trainer.diagnostics.items():
            print(f'| {k:25} | {v:25} |')

        if log_to_wandb:
            wandb.log(trainer.diagnostics)

        if save_models and ((t+1) % exp_args['save_models_every'] == 0 or
                            (t+1) == exp_args['num_iters']):
            with open(f'models/{run_name}.pt', 'wb') as f:
                state_dict = dict(model=model.state_dict(), optim=trainer.optim.state_dict())
                torch.save(state_dict, f)
            print(f'Saved model at {t+1} iters: {run_name}')

    if run is not None:
        run.finish()


def run_experiment(
        exp_name,
        experiment_params,
):
    exp_args = dict(
        num_iters = 1000,
        steps_per_iter = 100,
        test_steps_per_iter = 25,
        log_to_wandb = False,
        note = "",
        wandb_project = "mlnn_research_paper",
        include_date=False,
        save_models=False,
        save_models_every=250,
        device="cuda",
        gpu_batch_size=16
    )

    experiment_params['exp_name'] = exp_name
    experiment_params['exp_args'] = exp_args

    experiment(xp_name=exp_name, **experiment_params)
