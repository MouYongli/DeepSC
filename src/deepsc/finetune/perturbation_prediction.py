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
                    target_gene_values = batch.y
                    self.name2col = {g: i for i, g in enumerate(self.original_genes)}
                    pad_token_id = self.vocab[self.pad_token]  # e.g. <pad> 的 token id
                    device = batch.x.device
                    B, N = batch_size, self.num_genes

                    # 确保 gene_ids 是 torch.long tensor
                    if not isinstance(self.gene_ids, torch.Tensor):
                        self.gene_ids = torch.as_tensor(
                            self.gene_ids, dtype=torch.long, device=device
                        )
                    else:
                        self.gene_ids = self.gene_ids.to(
                            device=device, dtype=torch.long
                        )

                    # ------- 构造每个样本的 perturb 列 one-hot (全基因空间) -------
                    name2col = getattr(self, "name2col", None)
                    if name2col is None:
                        self.name2col = {
                            g: i for i, g in enumerate(self.original_genes)
                        }
                        name2col = self.name2col

                    pert_flags_full = torch.zeros(B, N, device=device, dtype=torch.long)
                    for r, p in enumerate(batch.pert):
                        # 例如 "TP53+ctrl" 或 "EGFR+BRCA1"
                        for g in p.split("+"):
                            if g and g != "ctrl":
                                j = name2col.get(g, -1)
                                if j != -1:
                                    pert_flags_full[r, j] = 1

                    # ------- 逐细胞选列 + 采样 + 排序 -------
                    sel_ids_list = []  # 每个细胞选到的 raw 列索引 (不等长)
                    genes_tok_list = []  # 每个细胞对应的 vocab id 序列 (不等长)
                    inp_vals_list = []  # 每个细胞的输入表达 (不等长)
                    tgt_vals_list = []  # 每个细胞的目标表达 (不等长)
                    pert_flags_list = []  # 每个细胞的 perturb 标记 (不等长)

                    Lmax = 0
                    for i in range(B):
                        # 基于该细胞：非零表达 或 该细胞的扰动基因；同时要求 gene_ids!=0（在 vocab 中）
                        expr_nz_i = (ori_gene_values[i] != 0) & (self.gene_ids != 0)
                        pert_cols_i = pert_flags_full[i].bool() & (self.gene_ids != 0)
                        sel_mask_i = expr_nz_i | pert_cols_i
                        sel_idx_i = torch.nonzero(sel_mask_i, as_tuple=True)[0]  # (Li,)

                        # 若为空，至少保底加入所有扰动基因列（可能仍为空），或者随机挑几个非 pad 的基因
                        if sel_idx_i.numel() == 0:
                            if pert_cols_i.any():
                                sel_idx_i = torch.nonzero(pert_cols_i, as_tuple=True)[0]
                            else:
                                # 退路：在有效基因里随机挑最多 sequence_length 个
                                valid_cols = torch.nonzero(
                                    self.gene_ids != 0, as_tuple=True
                                )[0]
                                k = min(valid_cols.numel(), self.args.sequence_length)
                                if k > 0:
                                    perm = torch.randperm(
                                        valid_cols.numel(), device=device
                                    )[:k]
                                    sel_idx_i = valid_cols[perm]

                        # 限长：优先保留扰动基因，再随机补足
                        Llim = self.args.sequence_length
                        if sel_idx_i.numel() > Llim:
                            # 先保留 perturb 列
                            pert_sel = torch.nonzero(
                                pert_cols_i[sel_idx_i], as_tuple=True
                            )[0]
                            non_pert_sel = torch.arange(
                                sel_idx_i.numel(), device=device
                            )
                            non_pert_sel = non_pert_sel[
                                ~torch.isin(non_pert_sel, pert_sel)
                            ]
                            keep = []
                            if pert_sel.numel() > 0:
                                # 如果扰动列多于限长，也只留前 Llim 个（或随机抽 Llim 个）
                                if pert_sel.numel() >= Llim:
                                    perm = torch.randperm(
                                        pert_sel.numel(), device=device
                                    )[:Llim]
                                    keep = pert_sel[perm]
                                else:
                                    keep = pert_sel
                                    need = Llim - pert_sel.numel()
                                    if non_pert_sel.numel() > 0 and need > 0:
                                        perm = torch.randperm(
                                            non_pert_sel.numel(), device=device
                                        )[:need]
                                        keep = torch.cat(
                                            [keep, non_pert_sel[perm]], dim=0
                                        )
                            else:
                                perm = torch.randperm(sel_idx_i.numel(), device=device)[
                                    :Llim
                                ]
                                keep = perm
                            sel_idx_i = sel_idx_i[keep]

                        # 按 vocab id 升序排序（稳定顺序）
                        genes_i = self.gene_ids[sel_idx_i]  # (Li,)
                        ord_i = torch.argsort(genes_i)
                        sel_idx_i = sel_idx_i[ord_i]
                        genes_i = genes_i[ord_i]

                        # 收集不等长序列
                        sel_ids_list.append(sel_idx_i)
                        genes_tok_list.append(genes_i)
                        inp_vals_list.append(ori_gene_values[i, sel_idx_i])
                        tgt_vals_list.append(target_gene_values[i, sel_idx_i])
                        pert_flags_list.append(pert_flags_full[i, sel_idx_i])

                        Lmax = max(Lmax, sel_idx_i.numel())

                    # ------- 把不等长序列 pad 成等长，并构造 padding mask -------
                    def pad1d(x, L, pad_val):
                        if x.numel() == L:
                            return x
                        out = x.new_full((L,), pad_val)
                        out[: x.numel()] = x
                        return out

                    mapped_input_gene_ids = torch.stack(
                        [pad1d(t, Lmax, pad_token_id) for t in genes_tok_list], dim=0
                    )  # (B, Lmax)
                    input_values = torch.stack(
                        [pad1d(t, Lmax, 0.0) for t in inp_vals_list], dim=0
                    ).to(
                        dtype=ori_gene_values.dtype
                    )  # (B, Lmax)
                    target_values = torch.stack(
                        [pad1d(t, Lmax, 0.0) for t in tgt_vals_list], dim=0
                    ).to(
                        dtype=target_gene_values.dtype
                    )  # (B, Lmax)
                    input_pert_flags = torch.stack(
                        [pad1d(t, Lmax, 0) for t in pert_flags_list], dim=0
                    ).to(
                        dtype=torch.long
                    )  # (B, Lmax)
                    print(input_values.shape)
                    print(target_values.shape)
                    print(input_pert_flags.shape)
                    print(mapped_input_gene_ids.shape)
                    # padding mask: True 表示是 pad 位置需要 mask
                    src_key_padding_mask = (
                        mapped_input_gene_ids == pad_token_id
                    )  # (B, Lmax)
                    # ------- 按 collator 的分箱规则对 input_values 做离散化（5 bins） -------
                    num_bins = 5
                    discrete_input_bins = torch.zeros_like(
                        input_values, dtype=torch.long
                    )
                    for i in range(B):
                        row_vals = input_values[i]
                        min_val = row_vals.min()
                        max_val = row_vals.max()
                        norm = (row_vals - min_val) / (max_val - min_val + 1e-8)
                        bins = torch.floor(norm * (num_bins - 1)).long()
                        bins = torch.clamp(bins, 0, num_bins - 1) + 1
                        discrete_input_bins[i] = bins
                    print(torch.unique(discrete_input_bins, return_counts=True))
                    self.model.train()
                    regression_output, y, gene_emb, expr_emb = self.model(
                        gene_ids=mapped_input_gene_ids,
                        expression_bin=discrete_input_bins,
                        normalized_expr=input_values,
                        input_pert_flags=input_pert_flags,
                    )
                    print(regression_output)
                    print(regression_output.shape)
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
