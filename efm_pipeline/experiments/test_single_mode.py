import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))
import torch
from efm_pipeline.efm_model import EFM_Tiny
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.train_efm import train
from efm_pipeline.utils import ensure_dir

class SingleModeDataset(OrderedAssemblyDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        L = self.embeddings.shape[1]
        M = L // self.W
        num_keys = (self.W - 1) * M
        self.token_ids[:, :num_keys] = torch.arange(num_keys, device=self.token_ids.device)
        for b in range(len(self.token_ids)):
            for i in range(num_keys, L):
                w = self.wave_ids[b, i].item()
                self.token_ids[b, i] = self.token_ids[b, (w - 1) * M + (i - num_keys)]
        rng = torch.Generator().manual_seed(43)
        self.embed_table = torch.randn(256, 512, generator=rng)
        self.embeddings = self.embed_table[self.token_ids]

dataset = SingleModeDataset(num_samples=1000, seq_len=64, vocab_size=256, embed_dim=512, W=4, seed=42)
model = EFM_Tiny(vocab_size=256, local_time_mode='continuous').cuda()
ensure_dir('tiny_single_mode/ckpt/tiny')
print('Training Single-Mode dataset...')
train(model=model, dataset=dataset, device=torch.device('cuda'), max_steps=1000, batch_size=64, save_every=1000, log_every=100, checkpoint_dir='tiny_single_mode/ckpt/tiny', experiment_name='tiny_sm', output_dir='tiny_single_mode', local_time_training='expansion', use_amp=True)
