
from model import BayesianFlowNetwork
from llama2 import BFNTransformer, ModelArgs
from tinystories import PretokDataset
import wandb
import torch
from composer import Trainer
from composer.models import ComposerModel
from composer.algorithms import GradientClipping
from composer.loggers import WandBLogger
from tokenizer import Tokenizer

# Lil Training Module Wrap for training with Composer 
class BFLlama(ComposerModel):
    def __init__(self, config) -> None:
        super().__init__()
        self.bfn = BayesianFlowNetwork(model=BFNTransformer(config), D=config.max_seq_len, K=config.vocab_size, beta=1.0)
        self.tokenizer = Tokenizer()
        self.internal_step = 0

    def forward(self, batch):
        x, y = batch
        bfn_loss = self.bfn.process(x)
        self.internal_step += 1

        if self.internal_step % 100 == 0:
                
            result = self.sample()[0].detach().cpu().tolist()
            print(self.tokenizer.decode(result))

        return {
            "bfn_loss": bfn_loss,
        }

    def loss(self, outputs, batch):
        return outputs["bfn_loss"]
    
    def sample(self, batch_size: int = 1, nb_steps: int = 100, device: str = 'cuda'):
        return self.bfn.sample(batch_size=batch_size, nb_steps=nb_steps, device=device)




if __name__ == "__main__":
    wandb_logger = WandBLogger(name="BFN-LLAMA2-TinyStories", project="BFN-LLAMA2")
    # 15M Parameters
    model_arguments = ModelArgs(
        dim= 288,
        n_layers=6,
        n_heads = 6,
        n_kv_heads=6, 
        vocab_size = 32000,
        multiple_of = 32,  # MLP hidden layer size will be multiple of
        norm_eps = 1e-5,
        max_seq_len = 256,
        dropout = 0.0,
    )

    model = BFLlama(model_arguments).cuda()
    lr = 5e-4
    betas = (0.9, 0.95)
    weight_decay = 0.00
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=True)
    gc = GradientClipping(clipping_type="norm", clipping_threshold=1.0)
    ds_train = PretokDataset(
        max_seq_len=256,
        split='train',
        vocab_size=model_arguments.vocab_size,
        vocab_source='llama2',
    )
    train_dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=512,
        pin_memory=True,
        num_workers=0,
    )


    ds_eval = PretokDataset(
        max_seq_len=256,
        split='eval',
        vocab_size=model_arguments.vocab_size,
        vocab_source='llama2',
    )
    eval_dataloader = torch.utils.data.DataLoader(
        ds_eval,
        batch_size=16,
        pin_memory=True,
        num_workers=0,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=[optimizer],
        algorithms=[gc],
        loggers=[wandb_logger],
        save_interval="10000ba",
        max_duration="100000ba",
        eval_subset_num_batches=100,
        device_train_microbatch_size="auto",
    )

    trainer.fit()
