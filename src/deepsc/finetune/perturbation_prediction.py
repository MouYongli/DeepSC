import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gears import PertData
from scipy.stats import pearsonr
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm import tqdm

from datetime import datetime
from deepsc.utils import (
    build_gene_ids_for_dataset,
    build_vocab_from_csv,
    extract_state_dict,
    report_loading_result,
    sample_weight_norms,
)
from src.deepsc.utils import (
    CosineAnnealingWarmRestartsWithDecayAndLinearWarmup,
    compute_perturbation_metrics,
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
        if args.pretrained_model:
            self.load_pretrained_model()
        self.optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.init_loss_fn()
        self.scheduler = self.create_scheduler(self.optimizer, self.args)
        self.vocab, self.id2vocab, self.pad_token, self.pad_value = (
            build_vocab_from_csv(self.args.csv_path)
        )
        self.prepare_data()
        self.gene_ids = build_gene_ids_for_dataset(self.original_genes, self.vocab)
        self.name2col = {g: i for i, g in enumerate(self.original_genes)}
        self.valid_gene_mask = self.gene_ids != 0
        self.perts_to_plot = ["KCTD16+ctrl"]
        # 比较self.name2col和self.node_map是不是内容一样的字典
        if hasattr(self, "node_map"):
            name2col_equal = self.name2col == self.node_map
            if self.is_master:
                print("self.name2col == self.node_map:", name2col_equal)
        else:
            if self.is_master:
                print("self does not have attribute 'node_map'")
        # print(torch.unique(self.gene_ids, return_counts=True))
        # unique_vals, counts = np.unique(self.gene_ids, return_counts=True)
        # print("unique_vals:",unique_vals)
        # print("counts:",counts)

    def init_loss_fn(self):
        self.criterion_mse = nn.MSELoss()

    def create_scheduler(self, optimizer, args):

        total_steps = args.epoch * math.ceil(
            (100000) / (args.batch_size * self.world_size * args.grad_acc)
        )
        warmup_ratio = self.args.warmup_ratio
        warmup_steps = math.ceil(total_steps * warmup_ratio)
        main_steps = total_steps - warmup_steps
        linear_warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_anneal = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
            optimizer,
            T_0=warmup_steps * 3,
            T_mult=1,
            warmup_steps=0,
            decay=0.9,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_warmup, cosine_anneal],
            milestones=[warmup_steps],
        )
        return scheduler

    def prepare_data(self):
        data_name = "adamson"
        split = "simulation"
        pert_data = PertData("./data")  # TODO: change to the data path
        self.pert_data = pert_data
        pert_data.load(data_name=data_name)
        pert_data.prepare_split(split=split, seed=1)
        pert_data.get_dataloader(
            batch_size=self.args.batch_size, test_batch_size=self.args.batch_size
        )
        self.original_genes = pert_data.adata.var["gene_name"].tolist()
        self.num_genes = len(self.original_genes)

        self.train_loader = pert_data.dataloader["train_loader"]
        self.valid_loader = pert_data.dataloader["val_loader"]
        self.train_loader, self.valid_loader = self.fabric.setup_dataloaders(
            self.train_loader, self.valid_loader
        )
        self.node_map = pert_data.node_map

    def _construct_pert_flags(self, data, batch_size, device):
        """构造每个样本的扰动标记矩阵"""
        pert_flags_full = torch.zeros(
            batch_size, self.num_genes, device=device, dtype=torch.long
        )

        # 标记原始扰动基因 (pert_flags=1)
        for r, p in enumerate(data.pert):
            for g in p.split("+"):
                if g and g != "ctrl":
                    j = self.name2col.get(g, -1)
                    if j != -1:
                        pert_flags_full[r, j] = 1

        # 标记受扰动影响较大的基因 (pert_flags=2)
        for r, p in enumerate(data.de_idx):
            for g_idx in p:
                pert_flags_full[r, g_idx] = 2

        return pert_flags_full

    def _select_genes_per_cell(self, ori_gene_values, pert_flags_full, device):
        """逐细胞选择基因并排序"""
        batch_size = ori_gene_values.shape[0]
        sel_ids_list = []
        genes_tok_list = []
        inp_vals_list = []
        tgt_vals_list = []
        pert_flags_list = []

        for i in range(batch_size):
            # 基于该细胞：非零表达 或 该细胞的扰动基因；同时要求 gene_ids!=0（在 vocab 中）
            expr_nz_i = (ori_gene_values[i] != 0) & (self.gene_ids != 0)
            pert_cols_i = pert_flags_full[i].bool() & (self.gene_ids != 0)
            sel_mask_i = expr_nz_i | pert_cols_i
            sel_idx_i = torch.nonzero(sel_mask_i, as_tuple=True)[
                0
            ]  # 如果不加[0]的话，是一个元组 得到的是一个数组，每个元素都是原始的基因id，包含的是哪些基因呢？有表达值且在我们的csv中 或被扰动且在我们的csv中

            # 若为空，至少保底加入所有扰动基因列（可能仍为空），或者随机挑几个非 pad 的基因
            if sel_idx_i.numel() == 0:
                if pert_cols_i.any():
                    sel_idx_i = torch.nonzero(pert_cols_i, as_tuple=True)[0]
                else:
                    # 退路：在有效基因里随机挑最多 sequence_length 个
                    valid_cols = torch.nonzero(self.gene_ids != 0, as_tuple=True)[0]
                    k = min(valid_cols.numel(), self.args.sequence_length)
                    if k > 0:
                        perm = torch.randperm(valid_cols.numel(), device=device)[:k]
                        sel_idx_i = valid_cols[perm]

            # 限长：优先保留扰动基因，再随机补足
            sel_idx_i = self._limit_sequence_length(sel_idx_i, pert_cols_i, device)

            # 按 vocab id 升序排序（稳定顺序）
            genes_i = self.gene_ids[sel_idx_i]
            ord_i = torch.argsort(genes_i)
            sel_idx_i = sel_idx_i[ord_i]
            genes_i = genes_i[ord_i]

            # 收集不等长序列
            sel_ids_list.append(sel_idx_i)
            genes_tok_list.append(genes_i)
            inp_vals_list.append(ori_gene_values[i, sel_idx_i])
            pert_flags_list.append(pert_flags_full[i, sel_idx_i])
        return sel_ids_list, genes_tok_list, inp_vals_list, pert_flags_list

    def _limit_sequence_length(self, sel_idx_i, pert_cols_i, device):
        """限制序列长度，优先保留扰动基因"""
        Llim = self.args.sequence_length
        if sel_idx_i.numel() <= Llim:
            return sel_idx_i

        # 先保留 perturb 列
        pert_sel = torch.nonzero(pert_cols_i[sel_idx_i], as_tuple=True)[0]
        non_pert_sel = torch.arange(sel_idx_i.numel(), device=device)
        non_pert_sel = non_pert_sel[~torch.isin(non_pert_sel, pert_sel)]

        if pert_sel.numel() > 0:
            if pert_sel.numel() >= Llim:
                perm = torch.randperm(pert_sel.numel(), device=device)[:Llim]
                keep = pert_sel[perm]
            else:
                keep = pert_sel
                need = Llim - pert_sel.numel()
                if non_pert_sel.numel() > 0 and need > 0:
                    perm = torch.randperm(non_pert_sel.numel(), device=device)[:need]
                    keep = torch.cat([keep, non_pert_sel[perm]], dim=0)
        else:
            perm = torch.randperm(sel_idx_i.numel(), device=device)[:Llim]
            keep = perm

        return sel_idx_i[keep]

    def _select_genes_scgpt_style(self, ori_gene_values, pert_flags_full, device):
        """采用scGPT式的基因选择策略：所有基因或随机采样"""
        batch_size = ori_gene_values.shape[0]

        # 选择所有有效基因（在vocab中的基因）
        valid_gene_mask = self.gene_ids != 0
        all_valid_gene_ids = torch.nonzero(valid_gene_mask, as_tuple=True)[0]

        # 如果基因数量超过序列长度限制，随机采样
        max_seq_len = getattr(self.args, "sequence_length", 1500)
        if len(all_valid_gene_ids) > max_seq_len:
            # 随机采样到max_seq_len
            perm = torch.randperm(len(all_valid_gene_ids), device=device)[:max_seq_len]
            selected_gene_ids = all_valid_gene_ids[perm]
        else:
            selected_gene_ids = all_valid_gene_ids

        # 按vocab id排序（保持一致性）
        genes_vocab_ids = self.gene_ids[selected_gene_ids]
        sort_idx = torch.argsort(genes_vocab_ids)
        selected_gene_ids = selected_gene_ids[sort_idx]
        genes_vocab_ids = genes_vocab_ids[sort_idx]

        # 所有样本使用相同的基因集合
        sel_ids_list = [selected_gene_ids for _ in range(batch_size)]
        genes_tok_list = [genes_vocab_ids for _ in range(batch_size)]
        inp_vals_list = [
            ori_gene_values[i, selected_gene_ids] for i in range(batch_size)
        ]
        pert_flags_list = [
            pert_flags_full[i, selected_gene_ids] for i in range(batch_size)
        ]

        return sel_ids_list, genes_tok_list, inp_vals_list, pert_flags_list

    def _pad_sequences(
        self,
        genes_tok_list,
        inp_vals_list,
        pert_flags_list,
        ori_gene_values,
        target_gene_values,
        sel_ids_list,
        pad_token_id,
    ):
        """将不等长序列填充为等长序列"""

        def pad1d(x, L, pad_val):
            if x.numel() == L:
                return x
            elif x.numel() > L:
                # 如果序列太长，截断到目标长度
                return x[:L]
            else:
                # 如果序列太短，进行padding
                out = x.new_full((L,), pad_val)
                out[: x.numel()] = x
                return out

        # 固定序列长度为1500，确保所有批次长度一致
        Lmax = 1500

        mapped_input_gene_ids = torch.stack(
            [pad1d(t, Lmax, pad_token_id) for t in genes_tok_list], dim=0
        )

        input_values = torch.stack(
            [pad1d(t, Lmax, -1) for t in inp_vals_list], dim=0
        ).to(dtype=ori_gene_values.dtype)

        # 重新构建target_values
        tgt_vals_list = []
        for i, sel_idx_i in enumerate(sel_ids_list):
            tgt_vals_list.append(target_gene_values[i, sel_idx_i])

        target_values = torch.stack(
            [pad1d(t, Lmax, -1) for t in tgt_vals_list], dim=0
        ).to(dtype=target_gene_values.dtype)

        # Add 1 to all values before padding (pert_flags: 0->1, 1->2, 2->3)
        pert_flags_list = [t + 1 for t in pert_flags_list]
        input_pert_flags = torch.stack(
            [pad1d(t, Lmax, 0) for t in pert_flags_list], dim=0
        ).to(dtype=torch.long)

        # padding mask: True 表示是 pad 位置需要 mask
        src_key_padding_mask = mapped_input_gene_ids == pad_token_id

        return (
            mapped_input_gene_ids,
            input_values,
            target_values,
            input_pert_flags,
            src_key_padding_mask,
        )

    def _discretize_expression(self, input_values, num_bins=5):
        """对表达值进行离散化分箱"""
        batch_size = input_values.shape[0]
        discrete_input_bins = torch.zeros_like(input_values, dtype=torch.long)

        for i in range(batch_size):
            row_vals = input_values[i]
            valid_mask = row_vals != -1.0
            if valid_mask.any():
                valid_vals = row_vals[valid_mask]
                min_val = valid_vals.min()
                max_val = valid_vals.max()
                norm = (valid_vals - min_val) / (max_val - min_val + 1e-8)
                bins = torch.floor(norm * (num_bins - 1)).long()
                bins = torch.clamp(bins, 0, num_bins - 1) + 1
                discrete_input_bins[i][valid_mask] = bins

        return discrete_input_bins

    def _preprocess_batch(self, data):
        """批量数据预处理的主函数"""
        batch_size = len(data.y)
        x: torch.Tensor = data.x
        ori_gene_values = x[:, 0].view(batch_size, self.num_genes)
        target_gene_values = data.y
        device = data.x.device
        pad_token_id = self.vocab[self.pad_token]

        # 确保 gene_ids 是正确的 tensor
        if not isinstance(self.gene_ids, torch.Tensor):
            self.gene_ids = torch.as_tensor(
                self.gene_ids, dtype=torch.long, device=device
            )
        else:
            self.gene_ids = self.gene_ids.to(device=device, dtype=torch.long)

        # 步骤1: 构造扰动标记
        pert_flags_full = self._construct_pert_flags(data, batch_size, device)

        # 步骤2: 采用scGPT式的基因选择策略
        if hasattr(self.args, "use_scgpt_selection") and self.args.use_scgpt_selection:
            # scGPT式：所有基因或随机采样
            sel_ids_list, genes_tok_list, inp_vals_list, pert_flags_list = (
                self._select_genes_scgpt_style(ori_gene_values, pert_flags_full, device)
            )
        else:
            # 原始方式：逐细胞选择基因
            sel_ids_list, genes_tok_list, inp_vals_list, pert_flags_list = (
                self._select_genes_per_cell(ori_gene_values, pert_flags_full, device)
            )
        # 在这之后还有可能出现基因表达值为0的基因，这不是padding，这是基因本身表达值为0
        # all_vals = torch.cat(inp_vals_list, dim=0)  # 将所有细胞的值合并
        # zero_count = (all_vals == 0).sum().item()
        # print(f"Zero elements before padding: {zero_count}")
        # print(f"inp_vals_list range: [{all_vals.min().item():.4f}, {all_vals.max().item():.4f}]")
        # 步骤3: 序列填充
        (
            mapped_input_gene_ids,
            input_values,
            target_values,
            input_pert_flags,
            src_key_padding_mask,
        ) = self._pad_sequences(
            genes_tok_list,
            inp_vals_list,
            pert_flags_list,
            ori_gene_values,
            target_gene_values,
            sel_ids_list,
            pad_token_id,
        )

        # 步骤4: 表达值离散化
        discrete_input_bins = self._discretize_expression(input_values)

        return {
            "mapped_input_gene_ids": mapped_input_gene_ids,
            "discrete_input_bins": discrete_input_bins,
            "input_values": input_values,
            "target_values": target_values,
            "input_pert_flags": input_pert_flags,
            "src_key_padding_mask": src_key_padding_mask,
            "sel_ids_list": sel_ids_list,  # 添加选择的基因索引信息
        }

    def each_training_iteration(self, data, is_accumulating):

        processed_data = self._preprocess_batch(data)

        # 解包预处理结果
        mapped_input_gene_ids = processed_data["mapped_input_gene_ids"]
        discrete_input_bins = processed_data["discrete_input_bins"]
        input_values = processed_data["input_values"]
        target_values = processed_data["target_values"]

        # input_pad_mask = (input_values == -1.0)
        # target_pad_mask = (target_values == -1.0)
        # mismatch = input_pad_mask != target_pad_mask

        # if mismatch.any():
        #     mismatch_indices = torch.nonzero(mismatch, as_tuple=False)
        #     print(f"First few mismatched positions: {mismatch_indices[:5]}")
        #     # 查看具体值
        #     for idx in mismatch_indices[:3]:
        #         batch_idx, seq_idx = idx[0].item(), idx[1].item()
        #         print(f"Batch {batch_idx}, Pos {seq_idx}:
        # input={input_values[batch_idx, seq_idx].item():.4f},
        #  target={target_values[batch_idx, seq_idx].item():.4f}")

        input_pert_flags = processed_data["input_pert_flags"]
        regression_output, y, gene_emb, expr_emb = self.model(
            gene_ids=mapped_input_gene_ids,
            expression_bin=discrete_input_bins,
            normalized_expr=input_values,
            input_pert_flags=input_pert_flags,
        )

        # 采用scGPT式损失计算：对所有非padding位置统一计算损失
        if hasattr(self.args, "use_scgpt_loss") and self.args.use_scgpt_loss:
            # scGPT式：所有有效位置都参与损失计算（不区分基因类型）
            valid_mask = (input_values != -1.0) & (target_values != -1.0)
            if valid_mask.any():
                # 统一MSE损失：所有基因一视同仁
                unified_loss = self.criterion_mse(
                    regression_output[valid_mask], target_values[valid_mask]
                )
                # 为了兼容原始接口，返回相同的损失值
                loss = unified_loss
                highly_affected_loss = unified_loss  # 实际上没有单独的"高影响"损失
                other_loss = unified_loss  # 实际上没有单独的"其他"损失
            else:
                # 没有有效数据的边界情况
                zero_loss = torch.tensor(0.0, device=regression_output.device)
                loss = zero_loss
                highly_affected_loss = zero_loss
                other_loss = zero_loss
        else:
            # 原始方式：加权损失计算
            # input_pert_flags的值：0=pad, 1=原始扰动基因, 2=受影响基因(DE), 3=受影响基因+1
            highly_affected_mask = (
                input_pert_flags == 3
            )  # pert_flags_full=2的基因加1后变成3
            other_mask = (input_pert_flags != 0) & (
                ~highly_affected_mask
            )  # 非pad且非高影响的基因
            valid_mask = (input_values != -1.0) & (target_values != -1.0)
            highly_affected_mask = highly_affected_mask & valid_mask
            other_mask = other_mask & valid_mask

            # 计算受扰动影响较大基因的MSE
            if highly_affected_mask.any():
                highly_affected_loss = self.criterion_mse(
                    regression_output[highly_affected_mask],
                    target_values[highly_affected_mask],
                )
            else:
                highly_affected_loss = torch.tensor(
                    0.0, device=regression_output.device
                )

            # 计算其他基因的MSE
            if other_mask.any():
                other_loss = self.criterion_mse(
                    regression_output[other_mask], target_values[other_mask]
                )
            else:
                other_loss = torch.tensor(0.0, device=regression_output.device)

            if valid_mask.any():
                valid_loss = self.criterion_mse(
                    regression_output[valid_mask], target_values[valid_mask]
                )
            else:
                valid_loss = torch.tensor(0.0, device=regression_output.device)

            loss = valid_loss

        if is_accumulating:
            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                self.fabric.backward(loss / self.args.grad_acc)
        else:
            self.fabric.backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
            self.optimizer.step()
            self.scheduler.step()  # 每次optimizer.step()后更新学习率
            self.optimizer.zero_grad()
        return loss, other_loss, highly_affected_loss

    def train(self):
        for epoch in range(self.args.epoch):
            self.model.train()
            data_iter = self.train_loader
            if self.is_master:
                data_iter = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch} [Finetune Cell Type Annotation]",
                    ncols=150,
                    position=1,
                )
            for index, batch in enumerate(data_iter):
                loss, other_loss, highly_affected_loss = self.each_training_iteration(
                    batch, is_accumulating=False
                )
                if self.is_master:
                    data_iter.set_postfix(
                        loss=loss.item(),
                        other_loss=other_loss.item(),
                        highly_affected_loss=highly_affected_loss.item(),
                    )
            self.res = self.eval(epoch)
            val_metrics = compute_perturbation_metrics(
                self.res,
                self.pert_data.adata[self.pert_data.adata.obs["condition"] == "ctrl"],
            )
            print(f"val_metrics at epoch {epoch}: ")
            print(val_metrics)

    def eval(self, epoch):
        self.model.eval()
        pert_cat = []
        pred = []
        truth = []
        pred_de = []
        truth_de = []
        results = {}
        data_iter = self.valid_loader
        if self.is_master:
            data_iter = tqdm(
                self.valid_loader,
                desc="EVAL ",
                ncols=150,
                position=1,
            )
        with torch.no_grad():
            for index, batch in enumerate(data_iter):
                pert_cat.extend(batch.pert)
                t = batch.y
                preprocessed_data = self._preprocess_batch(batch)
                mapped_input_gene_ids = preprocessed_data["mapped_input_gene_ids"]
                discrete_input_bins = preprocessed_data["discrete_input_bins"]
                input_values = preprocessed_data["input_values"]
                target_values = preprocessed_data["target_values"]
                input_pert_flags = preprocessed_data["input_pert_flags"]
                sel_ids_list = preprocessed_data["sel_ids_list"]

                regression_output, y, gene_emb, expr_emb = self.model(
                    gene_ids=mapped_input_gene_ids,
                    expression_bin=discrete_input_bins,
                    normalized_expr=input_values,
                    input_pert_flags=input_pert_flags,
                )

                # 改进版：将部分预测结果映射回完整基因空间
                batch_size = regression_output.size(0)
                device = regression_output.device

                # 不再使用零矩阵，而是用输入表达值作为基础（未扰动基因保持原值）
                # 之前是全是0为初始化
                # full_pred_gene_values = torch.zeros(batch_size,
                # self.num_genes, device=device, dtype=regression_output.dtype)
                ori_gene_values = (
                    batch.x[:, 0].view(batch_size, self.num_genes).to(device)
                )
                full_pred_gene_values = ori_gene_values.clone().to(
                    dtype=regression_output.dtype
                )

                # 将每个样本的预测值映射到对应的基因位置
                for i in range(batch_size):
                    if len(sel_ids_list) > i and sel_ids_list[i].numel() > 0:
                        # 获取当前样本选择的基因索引
                        selected_gene_indices = sel_ids_list[i]
                        # 确保不超过预测结果的长度
                        valid_len = min(
                            len(selected_gene_indices), regression_output.size(1)
                        )
                        if valid_len > 0:
                            full_pred_gene_values[
                                i, selected_gene_indices[:valid_len]
                            ] = regression_output[i, :valid_len]

                pred.extend(full_pred_gene_values.cpu())
                truth.extend(t.cpu())  # t已经是完整基因维度

                # 处理DE基因（从完整空间中提取）
                highly_affected_mask = (
                    input_pert_flags == 3
                )  # pert_flags_full=2的基因加1后变成3
                valid_mask = (input_values != -1.0) & (target_values != -1.0)
                highly_affected_mask = highly_affected_mask & valid_mask

                # 从压缩空间中提取DE基因的预测值
                pred_de.extend(regression_output[highly_affected_mask].cpu())
                truth_de.extend(target_values[highly_affected_mask].cpu())
        # 在循环结束后处理结果
        results["pert_cat"] = np.array(pert_cat)
        pred = torch.stack(pred)
        truth = torch.stack(truth)
        results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
        results["truth"] = truth.detach().cpu().numpy().astype(np.float64)
        results["pred_de"] = (
            torch.stack(pred_de).detach().cpu().numpy().astype(np.float64)
        )
        results["truth_de"] = (
            torch.stack(truth_de).detach().cpu().numpy().astype(np.float64)
        )
        return results

    # def eval(self, epoch):
    #     self.model.eval()

    #     # 初始化数据收集列表
    #     pert_cat = []
    #     highly_affected_predictions = []
    #     highly_affected_targets = []
    #     highly_affected_inputs = []
    #     others_predictions = []
    #     others_targets = []
    #     others_inputs = []
    #     pred = []
    #     truth = []
    #     overall_inputs = []

    #     # 收集控制组数据用于计算pearson_delta
    #     control_expressions = {}  # 基因名 -> 控制组表达值列表
    #     all_predictions = {}  # 基因名 -> 预测值列表 (对应target)
    #     all_targets = {}  # 基因名 -> 真实值列表
    #     gene_masks = {}  # 基因名 -> mask类型列表 (highly_affected 或 others)

    #     data_iter = self.valid_loader  # 修复：使用valid_loader而不是eval_loader
    #     if self.is_master:
    #         data_iter = tqdm(
    #             self.valid_loader,
    #             desc="EVAL ",
    #             ncols=150,
    #             position=1,
    #         )

    #     with torch.no_grad():  # 添加no_grad上下文管理器
    #         for index, batch in enumerate(data_iter):
    #             pert_cat.extend(batch.pert)
    #             t = batch.y

    #             preprocessed_data = self._preprocess_batch(batch)
    #             mapped_input_gene_ids = preprocessed_data["mapped_input_gene_ids"]
    #             discrete_input_bins = preprocessed_data["discrete_input_bins"]
    #             input_values = preprocessed_data["input_values"]
    #             target_values = preprocessed_data["target_values"]
    #             input_pert_flags = preprocessed_data["input_pert_flags"]
    #             regression_output, y, gene_emb, expr_emb = self.model(
    #                 gene_ids=mapped_input_gene_ids,
    #                 expression_bin=discrete_input_bins,
    #                 normalized_expr=input_values,
    #                 input_pert_flags=input_pert_flags,
    #             )

    #             highly_affected_mask = (
    #                 input_pert_flags == 3
    #             )  # pert_flags_full=2的基因加1后变成3
    #             other_mask = (input_pert_flags != 0) & (
    #                 ~highly_affected_mask
    #             )  # 非pad且非高影响的基因
    #             valid_mask = (input_values != -1.0) & (target_values != -1.0)
    #             highly_affected_mask = highly_affected_mask & valid_mask
    #             other_mask = other_mask & valid_mask

    #             # 收集数据用于散点图
    #             if highly_affected_mask.any():
    #                 highly_affected_predictions.append(
    #                     regression_output[highly_affected_mask].cpu()
    #                 )
    #                 highly_affected_targets.append(
    #                     target_values[highly_affected_mask].cpu()
    #                 )
    #                 highly_affected_inputs.append(
    #                     input_values[highly_affected_mask].cpu()
    #                 )
    #             if other_mask.any():
    #                 others_predictions.append(regression_output[other_mask].cpu())
    #                 others_targets.append(target_values[other_mask].cpu())
    #                 others_inputs.append(input_values[other_mask].cpu())
    #             if valid_mask.any():
    #                 pred.append(regression_output[valid_mask].cpu())
    #                 truth.append(target_values[valid_mask].cpu())
    #                 overall_inputs.append(input_values[valid_mask].cpu())

    #         pred_delta_ha, true_delta_ha = self.calc_deltas(
    #             highly_affected_predictions,
    #             highly_affected_targets,
    #             highly_affected_inputs,
    #         )
    #         pred_delta_others, true_delta_others = self.calc_deltas(
    #             others_predictions, others_targets, others_inputs
    #         )
    #         pred_delta_all, true_delta_all = self.calc_deltas(
    #             pred, truth, overall_inputs
    #         )
    #         print("=== Pearson Delta Results ===")
    #         if pred_delta_ha is not None:
    #             corr_ha, p_ha = pearsonr(pred_delta_ha, true_delta_ha)
    #             print(f"Highly Affected Delta: {corr_ha:.4f} (p={p_ha:.4e})")

    #         if pred_delta_others is not None:
    #             corr_others, p_others = pearsonr(pred_delta_others, true_delta_others)
    #             print(f"Others Delta: {corr_others:.4f} (p={p_others:.4e})")

    #         if pred_delta_all is not None:
    #             corr_all, p_all = pearsonr(pred_delta_all, true_delta_all)
    #             print(f"Overall Delta: {corr_all:.4f} (p={p_all:.4e})")

    #         self.plot_scatter_plots(
    #             highly_affected_predictions,
    #             highly_affected_targets,
    #             highly_affected_inputs,
    #             others_predictions,
    #             others_targets,
    #             others_inputs,
    #             pred,
    #             truth,
    #             overall_inputs,
    #             epoch,
    #         )

    def calc_deltas(self, preds_list, targets_list, inputs_list):
        all_preds = torch.cat(preds_list, dim=0) if preds_list else torch.tensor([])
        all_targets = (
            torch.cat(targets_list, dim=0) if targets_list else torch.tensor([])
        )
        all_inputs = torch.cat(inputs_list, dim=0) if inputs_list else torch.tensor([])

        if len(all_preds) == 0:
            return None, None

        # 计算delta（扰动后 - 控制组）
        pred_delta = (all_preds - all_inputs).cpu().numpy()
        true_delta = (all_targets - all_inputs).cpu().numpy()

        return pred_delta, true_delta

    def plot_scatter_plots(
        self,
        highly_affected_predictions,
        highly_affected_targets,
        highly_affected_inputs,
        others_predictions,
        others_targets,
        others_inputs,
        pred,
        truth,
        overall_inputs,
        epoch,
    ):
        """
        为三组数据画散点图，同时显示普通Pearson和Pearson Delta
        """

        # 合并数据并计算两种相关系数
        def prepare_data_and_metrics(preds_list, targets_list, inputs_list):
            if not preds_list:
                return np.array([]), np.array([]), 0.0, 0.0

            all_preds = torch.cat(preds_list, dim=0).cpu().numpy()
            all_targets = torch.cat(targets_list, dim=0).cpu().numpy()
            all_inputs = torch.cat(inputs_list, dim=0).cpu().numpy()

            # 普通pearson
            pearson_normal, _ = pearsonr(all_preds, all_targets)

            # pearson delta
            pred_delta = all_preds - all_inputs
            target_delta = all_targets - all_inputs
            pearson_delta, _ = pearsonr(pred_delta, target_delta)

            return all_preds, all_targets, pearson_normal, pearson_delta

        # 准备三组数据
        ha_pred, ha_target, ha_pearson, ha_delta = prepare_data_and_metrics(
            highly_affected_predictions, highly_affected_targets, highly_affected_inputs
        )
        other_pred, other_target, other_pearson, other_delta = prepare_data_and_metrics(
            others_predictions, others_targets, others_inputs
        )
        all_pred, all_target, all_pearson, all_delta = prepare_data_and_metrics(
            pred, truth, overall_inputs
        )

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Epoch {epoch} - Prediction vs Target", fontsize=16)

        datasets = [
            (
                ha_pred,
                ha_target,
                ha_pearson,
                ha_delta,
                "Highly Affected Genes",
                "red",
                axes[0],
            ),
            (
                other_pred,
                other_target,
                other_pearson,
                other_delta,
                "Other Genes",
                "blue",
                axes[1],
            ),
            (
                all_pred,
                all_target,
                all_pearson,
                all_delta,
                "All Genes",
                "green",
                axes[2],
            ),
        ]

        for pred_data, target_data, pearson, delta, title, color, ax in datasets:
            if len(pred_data) > 0:
                # 画散点图
                ax.scatter(target_data, pred_data, alpha=0.6, s=20, color=color)

                # 添加对角线
                min_val = min(pred_data.min(), target_data.min())
                max_val = max(pred_data.max(), target_data.max())
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "k--",
                    alpha=0.8,
                    linewidth=1,
                )

                # 设置标签和标题（包含两个相关系数）
                ax.set_xlabel("True Expression")
                ax.set_ylabel("Predicted Expression")
                ax.set_title(
                    f"{title}\n"
                    f"Pearson: {pearson:.4f}\n"
                    f"Pearson Δ: {delta:.4f}\n"
                    f"n = {len(pred_data)}",
                    fontsize=10,
                )
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title)

        plt.tight_layout()

        # 保存文件
        # 创建带时间戳的文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = (
            f"/home/angli/baseline/DeepSC/results/perturbation_prediction_{timestamp}"
        )
        os.makedirs(results_dir, exist_ok=True)

        save_path = os.path.join(
            results_dir, f"perturbation_scatter_epoch_{epoch:03d}_model_epoch_3.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        if self.is_master:
            print(f"Scatter plots saved to: {save_path}")
            print(
                f"Metrics - HA: r={ha_pearson:.4f}, Δ={ha_delta:.4f} | "
                f"Others: r={other_pearson:.4f}, Δ={other_delta:.4f} | "
                f"All: r={all_pearson:.4f}, Δ={all_delta:.4f}"
            )

        return save_path

    def load_pretrained_model(self):
        ckpt_path = self.args.pretrained_model_path
        assert os.path.exists(ckpt_path), f"找不到 ckpt: {ckpt_path}"

        # 2) 只在 rank0 读取到 CPU，减少压力；再广播（可选）
        if self.fabric.global_rank == 0:
            print(f"[LOAD] 读取 checkpoint: {ckpt_path}")
            raw = torch.load(ckpt_path, map_location="cpu")
            state_dict = extract_state_dict(raw)
        else:
            raw = None
            state_dict = None

        # 3) 广播到所有进程
        state_dict = self.fabric.broadcast(state_dict, src=0)
        # 4) 打印抽样对比（可选，但很直观）
        sample_weight_norms(self.model, state_dict, k=5)

        # 5) 真正加载（strict=False：允许你新增的 embedding 层留空）
        load_info = self.model.load_state_dict(state_dict, strict=False)
        report_loading_result(load_info)
