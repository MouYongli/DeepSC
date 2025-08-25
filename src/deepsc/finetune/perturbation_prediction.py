import torch
from gears import PertData
from torch.optim import Adam

from deepsc.utils import build_gene_ids_for_dataset, build_vocab_from_csv
from src.deepsc.utils import (
    seed_all,
)


class PerturbationPrediction:
    # Question:在这个脚本中，基因的顺序为乱序，不确定这样会不会对结果有影响
    def __init__(self, args, fabric, model):
        self.args = args
        self.fabric = fabric
        self.model = model
        self.world_size = self.fabric.world_size
        seed_all(args.seed + self.fabric.global_rank)
        self.is_master = self.fabric.global_rank == 0
        self.optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.init_loss_fn()
        self.scheduler = self.create_scheduler(self.optimizer, self.args)
        self.vocab, self.id2vocab, self.pad_token, self.pad_value = (
            build_vocab_from_csv(self.args.csv_path)
        )
        self.prepare_data()
        self.gene_ids = build_gene_ids_for_dataset(self.original_genes, self.vocab)
        self.valid_gene_mask = self.gene_ids != 0
        # print(torch.unique(self.gene_ids, return_counts=True))
        # unique_vals, counts = np.unique(self.gene_ids, return_counts=True)
        # print("unique_vals:",unique_vals)
        # print("counts:",counts)

    def prepare_data(self):
        data_name = "adamson"
        split = "simulation"
        batch_size = 64
        eval_batch_size = 64
        pert_data = PertData("./data")  # TODO: change to the data path
        pert_data.load(data_name=data_name)
        pert_data.prepare_split(split=split, seed=1)
        pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
        self.original_genes = pert_data.adata.var["gene_name"].tolist()
        self.num_genes = len(self.original_genes)
        self.train_loader = pert_data.dataloader["train_loader"]
        self.valid_loader = pert_data.dataloader["val_loader"]
        # genes = pert_data.adata.var["gene_name"].tolist()
        # pad_token = "<pad>"
        # special_tokens = [pad_token, "<cls>", "<eoc>"]
        # vocab = Vocab(
        #     VocabPybind(genes + special_tokens, None)
        # )  # bidirectional lookup [gene <-> int]
        # vocab.set_default_index(vocab["<pad>"])
        # gene_ids = np.array(
        #     [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
        # )
        # self.n_genes = len(genes)

    def init_loss_fn(self):
        pass

    def create_scheduler(self, optimizer, args):
        pass

    def train(self):
        for epoch in range(self.args.epoch):
            for index, batch in enumerate(self.train_loader):
                if index == 1 and epoch == 1 and self.is_master:
                    batch_size = len(batch.y)
                    x: torch.Tensor = batch.x  # (batch_size * n_genes, 1)
                    ori_gene_values = x[:, 0].view(batch_size, self.num_genes)

                    self.name2col = {g: i for i, g in enumerate(self.original_genes)}

                    # 在 train 循环拿到 batch 后：
                    batch_size = len(batch.y)
                    pert_flags_full = torch.zeros(
                        batch_size,
                        self.num_genes,
                        device=batch.x.device,
                        dtype=torch.long,
                    )

                    for r, p in enumerate(batch.pert):
                        # p 例如: "TP53+ctrl" 或 "EGFR+BRCA1"
                        for g in p.split("+"):
                            if g == "ctrl" or g == "" or g is None:
                                continue
                            col = self.name2col.get(g, -1)
                            if col != -1:
                                pert_flags_full[r, col] = 1

                    target_gene_values = batch.y
                    expr_nonzero_any = (ori_gene_values != 0).any(dim=0)
                    both_ok_cols_mask = expr_nonzero_any & self.valid_gene_mask  # (N,)
                    input_gene_ids = torch.nonzero(both_ok_cols_mask, as_tuple=True)[
                        0
                    ]  # (L,)
                    # Question: 此处随机抽取，但是整个batch都是一样的。不知道会不会有问题。我觉得只要把sequence length设大就不会有问题
                    # TODO: 不能随机！因为会把pert给筛掉！！或者我们可以让perturb不能被筛选掉？？？？
                    if input_gene_ids.numel() > self.args.sequence_length:
                        perm = torch.randperm(
                            input_gene_ids.numel(), device=input_gene_ids.device
                        )
                        input_gene_ids = input_gene_ids[
                            perm[: self.args.sequence_length]
                        ]  # (max_seq_len,)

                    genes = self.gene_ids[input_gene_ids]
                    # 按基因大小重拍
                    sort_idx = torch.argsort(genes)
                    input_values = ori_gene_values[:, input_gene_ids]  # (B, L)
                    target_values = target_gene_values[:, input_gene_ids]
                    # c重拍基因名
                    genes = genes[sort_idx]
                    genes = genes.repeat(batch_size, 1)
                    input_values = input_values[:, sort_idx]
                    target_values = target_values[:, sort_idx]
                    pert_flags_full = pert_flags_full[:, input_gene_ids]
                    pert_flags_full = pert_flags_full[:, sort_idx]

                    # for i in range(batch_size):
                    #     print(batch.pert_idx[i])
                    #     print(torch.unique(pert_flags_full[i], return_counts=True))
                    # #print(pert_flags_full)
                    # print("the shape of input_values:",input_values.shape)
                    # print("the shape of target_values:",target_values.shape)

                    # ori_gene_values = x[:, 0].view(batch_size, self.n_genes)
                    # print(ori_gene_values.shape)
                pass

    def eval(self):
        for index, batch in enumerate(self.valid_loader):
            pass

    def test(self):
        for index, batch in enumerate(self.test_loader):
            pass
