from deepsc.models.deepsc.model import ExpressionEmbedding
import torch

x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
expr_emb = ExpressionEmbedding(embedding_dim=10, num_bins=50, alpha=0.1)
y = expr_emb(x)























class L0Gate(nn.Module):
    """
    L0正则化门控层，用于实现稀疏性
    
    实现基于L0正则化的稀疏门控机制，通过可学习的门控参数控制连接的开关
    """
    def __init__(self, input_dim: int, temperature: float = 0.1, hard: bool = True):
        """
        Args:
            input_dim: 输入维度
            temperature: Gumbel Softmax温度参数
            hard: 是否使用硬采样
        """
        super(L0Gate, self).__init__()
        self.temperature = temperature
        self.hard = hard
        
        # 门控参数：log(π/(1-π))，其中π是连接存在的概率
        self.gate_logits = nn.Parameter(torch.randn(input_dim))
        
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量, shape: (..., input_dim)
            training: 是否处于训练模式
            
        Returns:
            gated_x: 门控后的输出
            gate_probs: 门控概率，用于L0正则化计算
        """
        if training:
            # 训练时使用Gumbel Sigmoid采样
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.gate_logits) + 1e-8) + 1e-8)
            gate_logits = (self.gate_logits + gumbel_noise) / self.temperature
            gate_probs = torch.sigmoid(gate_logits)
            
            if self.hard:
                # 硬采样：直通估计器
                gate_hard = (gate_probs > 0.5).float()
                gate_probs = gate_hard - gate_probs.detach() + gate_probs
        else:
            # 推理时使用确定性阈值
            gate_probs = torch.sigmoid(self.gate_logits)
            gate_probs = (gate_probs > 0.5).float()
        
        # 应用门控
        gated_x = x * gate_probs.unsqueeze(0).expand_as(x)
        
        return gated_x, gate_probs
    
    def l0_regularization(self) -> torch.Tensor:
        """
        计算L0正则化项
        
        Returns:
            l0_reg: L0正则化损失
        """
        # 计算每个门的期望稀疏度
        gate_probs = torch.sigmoid(self.gate_logits)
        l0_reg = torch.sum(gate_probs)
        return l0_reg


class GumbellThreeWayClassifier(nn.Module):
    """
    基于Gumbel Softmax的三态分类器：抑制(-1)、无作用(0)、激活(+1)
    """
    def __init__(self, input_dim: int, temperature: float = 1.0, hard: bool = True):
        """
        Args:
            input_dim: 输入维度
            temperature: Gumbel Softmax温度参数
            hard: 是否使用硬采样
        """
        super(GumbellThreeWayClassifier, self).__init__()
        self.temperature = temperature
        self.hard = hard
        
        # 三态分类的logits：[抑制, 无作用, 激活]
        self.classification_logits = nn.Parameter(torch.randn(input_dim, 3))
        
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量, shape: (..., input_dim)
            training: 是否处于训练模式
            
        Returns:
            classified_x: 分类后的输出（-1, 0, +1）
            class_probs: 三态概率分布
        """
        batch_shape = x.shape[:-1]
        input_dim = x.shape[-1]
        
        if training:
            # 训练时使用Gumbel Softmax采样
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.classification_logits) + 1e-8) + 1e-8)
            logits = (self.classification_logits + gumbel_noise) / self.temperature
            class_probs = F.softmax(logits, dim=-1)  # shape: (input_dim, 3)
            
            if self.hard:
                # 硬采样：直通估计器
                class_hard = F.one_hot(torch.argmax(class_probs, dim=-1), num_classes=3).float()
                class_probs = class_hard - class_probs.detach() + class_probs
        else:
            # 推理时使用确定性分类
            class_probs = F.softmax(self.classification_logits, dim=-1)
            class_hard = F.one_hot(torch.argmax(class_probs, dim=-1), num_classes=3).float()
            class_probs = class_hard
        
        # 将one-hot编码转换为数值：[抑制=-1, 无作用=0, 激活=+1]
        class_values = torch.tensor([-1.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
        classified_values = torch.sum(class_probs * class_values.unsqueeze(0), dim=-1)  # shape: (input_dim,)
        
        # 应用分类结果到输入
        classified_x = x * classified_values.unsqueeze(0).expand_as(x)
        
        return classified_x, class_probs


class GeneRegulationEmbedding(nn.Module):
    """
    Gene Embedding分支：专注于捕捉基因的内在生物学特性和基因间的调控关系
    
    学习基因的语义表示，包括：
    - 功能相似性：功能相关的基因在嵌入空间中距离较近
    - 通路关系：同一生物学通路的基因具有相似的表示
    - 调控关系：转录因子与其靶基因之间的关系
    
    新增功能：
    - 基因调控网络(GRN)推断
    - L0正则化稀疏性控制
    - Gumbel Softmax三态调控关系分类
    """
    def __init__(self, 
                 num_genes: int, 
                 embedding_dim: int,
                 enable_grn: bool = True,
                 l0_temperature: float = 0.1,
                 gumbel_temperature: float = 1.0,
                 l0_lambda: float = 1e-3):
        """
        Args:
            num_genes: 基因数量 g
            embedding_dim: 嵌入维度 d
            enable_grn: 是否启用基因调控网络推断
            l0_temperature: L0正则化温度参数
            gumbel_temperature: Gumbel Softmax温度参数
            l0_lambda: L0正则化权重
        """
        super(GeneRegulationEmbedding, self).__init__()
        self.num_genes = num_genes
        self.embedding_dim = embedding_dim
        self.enable_grn = enable_grn
        self.l0_lambda = l0_lambda
        
        # 基因嵌入层：每个基因对应一个可学习的向量
        self.gene_embedding = nn.Embedding(
            num_embeddings=num_genes,
            embedding_dim=embedding_dim
        )
        
        # 初始化嵌入权重
        nn.init.xavier_uniform_(self.gene_embedding.weight)
        
        # 基因调控网络推断模块
        if self.enable_grn:
            # 相互作用强度计算层
            self.interaction_projector = nn.Linear(embedding_dim * 2, 1)
            
            # L0正则化门控层
            self.l0_gate = L0Gate(
                input_dim=num_genes * num_genes,  # 全连接的基因对
                temperature=l0_temperature,
                hard=True
            )
            
            # Gumbel三态分类器
            self.three_way_classifier = GumbellThreeWayClassifier(
                input_dim=num_genes * num_genes,
                temperature=gumbel_temperature,
                hard=True
            )
    
    def compute_gene_interactions(self, gene_embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算基因间的相互作用强度
        
        Args:
            gene_embeddings: 基因嵌入, shape: (batch_size, g, d)
            
        Returns:
            interaction_matrix: 相互作用强度矩阵, shape: (batch_size, g, g)
        """
        batch_size, num_genes, embed_dim = gene_embeddings.shape
        
        # 创建所有基因对的组合
        # gene_i: (batch_size, g, 1, d) -> (batch_size, g, g, d)
        gene_i = gene_embeddings.unsqueeze(2).expand(-1, -1, num_genes, -1)
        # gene_j: (batch_size, 1, g, d) -> (batch_size, g, g, d)
        gene_j = gene_embeddings.unsqueeze(1).expand(-1, num_genes, -1, -1)
        
        # 拼接基因对特征: (batch_size, g, g, 2*d)
        gene_pairs = torch.cat([gene_i, gene_j], dim=-1)
        
        # 计算相互作用强度: (batch_size, g, g, 1) -> (batch_size, g, g)
        interaction_scores = self.interaction_projector(gene_pairs).squeeze(-1)
        
        return interaction_scores
    
    def infer_grn(self, gene_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        推断基因调控网络
        
        Args:
            gene_embeddings: 基因嵌入, shape: (batch_size, g, d)
            
        Returns:
            grn_results: 包含调控网络信息的字典
                - interaction_matrix: 相互作用强度矩阵, shape: (batch_size, g, g)
                - sparse_matrix: L0稀疏化后的矩阵, shape: (batch_size, g, g)
                - regulatory_matrix: 三态调控矩阵(-1/0/+1), shape: (batch_size, g, g)
                - gate_probs: L0门控概率
                - class_probs: 三态分类概率
                - l0_reg: L0正则化损失
        """
        # 计算基因间相互作用强度
        interaction_matrix = self.compute_gene_interactions(gene_embeddings)  # (batch_size, g, g)
        
        # 展平为向量进行门控和分类
        batch_size = interaction_matrix.shape[0]
        flattened_interactions = interaction_matrix.view(batch_size, -1)  # (batch_size, g*g)
        
        # L0稀疏化
        sparse_interactions, gate_probs = self.l0_gate(
            flattened_interactions, training=self.training
        )
        
        # 三态分类
        regulatory_interactions, class_probs = self.three_way_classifier(
            sparse_interactions, training=self.training
        )
        
        # 重塑为矩阵形式
        sparse_matrix = sparse_interactions.view(batch_size, self.num_genes, self.num_genes)
        regulatory_matrix = regulatory_interactions.view(batch_size, self.num_genes, self.num_genes)
        
        # 计算L0正则化损失
        l0_reg = self.l0_gate.l0_regularization()
        
        return {
            'interaction_matrix': interaction_matrix,
            'sparse_matrix': sparse_matrix,
            'regulatory_matrix': regulatory_matrix,
            'gate_probs': gate_probs,
            'class_probs': class_probs,
            'l0_reg': l0_reg
        }
    
    def forward(self, gene_ids: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            gene_ids: 基因ID序列 G = [g_1, g_2, ..., g_g], shape: (batch_size, g)
            
        Returns:
            gene_embeddings: 基因嵌入 E_gene ∈ R^{g×d}, shape: (batch_size, g, d)
            grn_results: 基因调控网络推断结果（如果启用）
        """
        # E_gene = f_gene(G)
        gene_embeddings = self.gene_embedding(gene_ids)
        
        # 基因调控网络推断
        grn_results = None
        if self.enable_grn:
            grn_results = self.infer_grn(gene_embeddings)
        
        return gene_embeddings, grn_results





class DeepSC(nn.Module):
    """
    DeepSC模型主类
    
    整合Gene Embedding和Expression Embedding两个分支
    新增基因调控网络推断功能
    """
    def __init__(self, 
                 num_genes: int,
                 embedding_dim: int = 512,
                 num_bins: int = 50,
                 alpha: float = 0.1,
                 enable_grn: bool = True,
                 l0_temperature: float = 0.1,
                 gumbel_temperature: float = 1.0,
                 l0_lambda: float = 1e-3):
        """
        Args:
            num_genes: 基因数量
            embedding_dim: 嵌入维度
            num_bins: 表达量离散化的bin数量
            alpha: 连续特征权重参数
            enable_grn: 是否启用基因调控网络推断
            l0_temperature: L0正则化温度参数
            gumbel_temperature: Gumbel Softmax温度参数
            l0_lambda: L0正则化权重
        """
        super(DeepSC, self).__init__()
        self.num_genes = num_genes
        self.embedding_dim = embedding_dim
        self.enable_grn = enable_grn
        self.l0_lambda = l0_lambda
        
        # Gene Embedding分支
        self.gene_embedding = GeneRegulationEmbedding(
            num_genes=num_genes,
            embedding_dim=embedding_dim,
            enable_grn=enable_grn,
            l0_temperature=l0_temperature,
            gumbel_temperature=gumbel_temperature,
            l0_lambda=l0_lambda
        )
        
        # Expression Embedding分支
        self.expression_embedding = ExpressionEmbedding(
            embedding_dim=embedding_dim,
            num_bins=num_bins,
            alpha=alpha
        )

    def forward(self, gene_ids: torch.Tensor, expression: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            gene_ids: 基因ID序列, shape: (batch_size, g)
            expression: 表达量向量, shape: (batch_size, g)
            
        Returns:
            outputs: 包含所有输出的字典
                - gene_embeddings: 基因嵌入 E_gene, shape: (batch_size, g, d)
                - expr_embeddings: 表达量嵌入 E_expr, shape: (batch_size, g, d)
                - grn_results: 基因调控网络推断结果（如果启用）
        """
        # Gene Embedding: E_gene = f_gene(G)
        gene_embeddings, grn_results = self.gene_embedding(gene_ids)
        
        # Expression Embedding: E_expr = f_expr(x)
        expr_embeddings = self.expression_embedding(expression)
        
        # 整合输出
        outputs = {
            'gene_embeddings': gene_embeddings,
            'expr_embeddings': expr_embeddings
        }
        
        # 添加基因调控网络结果
        if grn_results is not None:
            outputs['grn_results'] = grn_results
        
        return outputs
    
    def compute_total_loss(self, outputs: Dict[str, torch.Tensor], 
                          main_loss: torch.Tensor) -> torch.Tensor:
        """
        计算总损失，包括主损失和L0正则化项
        
        Args:
            outputs: 模型输出
            main_loss: 主任务损失
            
        Returns:
            total_loss: 总损失
        """
        total_loss = main_loss
        
        # 添加L0正则化损失
        if self.enable_grn and 'grn_results' in outputs:
            l0_reg = outputs['grn_results']['l0_reg']
            total_loss = total_loss + self.l0_lambda * l0_reg
        
        return total_loss
    
    def get_regulatory_network(self, outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        获取推断的基因调控网络
        
        Args:
            outputs: 模型输出
            
        Returns:
            regulatory_matrix: 调控矩阵，值为-1(抑制)/0(无作用)/+1(激活)
        """
        if self.enable_grn and 'grn_results' in outputs:
            return outputs['grn_results']['regulatory_matrix']
        return None

    def train(self, train_data, val_data):
        """训练方法 - 待实现"""
        pass

    def predict(self, test_data):
        """预测方法 - 待实现"""
        pass
