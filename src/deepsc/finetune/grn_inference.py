import os
from collections import defaultdict

import anndata as ad
import pandas as pd
import scanpy as sc
import torch

from deepsc.scgpt_utils.grn import GeneEmbedding


class GRNInference:
    def __init__(self, args, fabric, model):
        self.args = args
        self.fabric = fabric
        self.model = model

        # 设备
        self.device = next(model.parameters()).device

        self.load_pretrained_model()

        # 1) 读 CSV，构建映射
        df = pd.read_csv(
            "/home/angli/baseline/DeepSC/scripts/data/preprocessing/gene_map.csv"
        )
        # feature_name -> id（允许重复 id）
        self.gene2idx = dict(zip(df["feature_name"], df["id"]))
        # id -> [feature_name, ...]（可选：用于反查）
        self.idx2genes = defaultdict(list)
        for name, i in zip(df["feature_name"], df["id"]):
            self.idx2genes[int(i)].append(name)

        # 2) 载入 AnnData
        self.load_data()

        # 3) 取嵌入
        self.get_gene_embeddings()
        embed = GeneEmbedding(self.gene2emb)

        # Perform Louvain clustering with desired resolution; here we specify resolution=40
        gdata = embed.get_adata(resolution=40)
        # Retrieve the gene clusters
        metagenes = embed.get_metagenes(gdata)

        # Obtain the set of gene programs from clusters with #genes >= 5
        mgs = dict()
        for mg, genes in metagenes.items():
            if len(genes) > 4:
                mgs[mg] = genes
        print(mgs)
        # sns.set(font_scale=0.35)
        # print("Scoring metagenes")
        # embed.score_metagenes(self.adata, metagenes)
        # print("Plotting metagenes scores")
        # embed.plot_metagenes_scores(self.adata, mgs, "celltype")
        # print("Saving metagenes scores")
        # plt.savefig("metagenes_scores.png", dpi=300, bbox_inches="tight")
        # plt.close()
        # print("Metagenes scores saved")

    def load_data(self):
        self.adata = ad.read_h5ad(self.args.data_path)
        self.adata.obs["celltype"] = self.adata.obs["final_annotation"].astype(str)
        sc.pp.highly_variable_genes(
            self.adata,
            layer=None,
            n_top_genes=1200 if isinstance(1200, int) else None,
            batch_key="batch",
            flavor="seurat_v3",
            subset=True,
        )
        print(self.adata.shape)

    def get_gene_embeddings(self):
        """
        目标：
        - 生成 2 个结构：
          a) self.gene_embeddings_mat: (N, D) 矩阵，顺序与 features_aligned 一致
          b) self.gene2emb: {gene_name -> embedding(np.ndarray)}
        其中 N 是 adata.var.index 中出现、且在 gene_map.csv 中有 id 的基因数。
        """
        # 仅保留在 adata 中出现的基因，并按照这些 gene 的顺序取 id
        adata_genes = set(map(str, self.adata.var.index))
        features_aligned = [g for g in self.gene2idx.keys() if g in adata_genes]

        # 用真正的 id 列表（而不是 enumerate 的序号）
        id_list = [int(self.gene2idx[g]) for g in features_aligned]

        # as_tensor + 直接放到 device；避免对 tensor 再次 torch.tensor(...) 的告警
        ids = torch.as_tensor(id_list, dtype=torch.long, device=self.device)

        # 取 embedding（(N, D)）
        with torch.no_grad():
            emb_mat = self.model.gene_embedding(ids)

        # 保存矩阵与字典（注意：名字不要再叫 gene_embeddings，避免冲突）
        self.gene_embeddings_mat = emb_mat.cpu().numpy()  # shape: (N, D)
        self.gene2emb = {
            g: self.gene_embeddings_mat[i]  # dict: gene -> np.ndarray(D,)
            for i, g in enumerate(features_aligned)
        }

        print(
            f"Retrieved gene embeddings for {len(self.gene2emb)} genes. "
            f"Matrix shape: {self.gene_embeddings_mat.shape}"
        )

        # （可选）如果还想拿“唯一 id 的嵌入”，再来一份去重版：
        unique_ids = sorted(set(id_list))
        unique_ids_t = torch.as_tensor(unique_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            emb_unique = self.model.gene_embedding(unique_ids_t).cpu().numpy()
        # id -> emb
        self.id2emb = {uid: emb_unique[j] for j, uid in enumerate(unique_ids)}

    def train(self):
        pass

    def load_pretrained_model(self):
        ckpt_path = self.args.pretrained_model_path
        assert os.path.exists(ckpt_path), f"找不到 ckpt: {ckpt_path}"
        if self.fabric.global_rank == 0:
            print(f"[LOAD] 读取 checkpoint: {ckpt_path}")
            raw = torch.load(ckpt_path, map_location="cpu")
            from deepsc.utils.utils import (
                extract_state_dict_with_encoder_prefix,
                report_loading_result,
                sample_weight_norms,
            )

            state_dict = extract_state_dict_with_encoder_prefix(raw)
        else:
            state_dict = None
        state_dict = self.fabric.broadcast(state_dict, src=0)
        sample_weight_norms(self.model, state_dict, k=5)
        load_info = self.model.load_state_dict(state_dict, strict=False)

        report_loading_result(load_info)
