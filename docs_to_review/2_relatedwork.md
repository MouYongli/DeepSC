# Related Work

To analyze and compare the ability of large-scale transformer-based pretrained models in scRNA-seq data analysis and researches, this study selects scGPT, scBERT, scFoundation, and Geneformer as baseline models. These models generally follow a common paradigm that includes data preprocessing, embedding generation, self-supervised pretraining, and task-specific fine-tuning to build models for downstream applications. They have demonstrated strong ability in transfer learning and generalization across various downstream tasks such as cell type annotation, perturbation prediction, and gene regulatory network inference.

## Baseline Models

### scBERT

scBERT is the earliest study to introduce a Transformer-based foundation model for scRNA-seq data analysis. During the pretraining stage, scBERT was trained on over 1.1 million human single-cell transcriptomes from the PanglaoDB database, which covers 74 tissues and 209 datasets. The model performs log-normalization with a size factor of 10,000 and filters out low-quality cells with fewer than 200 detected genes. Unlike traditional pipelines that rely on highly variable gene selection or dimensionality reduction, scBERT takes all genes (>20,000) as input of the model to preserve gene-level interpretability. The expression values are discretized through a term-frequency-based binning strategy.

scBERT adopts the Performer architecture as its encoder backbone, which improves computational efficiency and scalability when processing extremely long gene sequences. During the encoding stage, each gene token is first embedded using `gene2vec` to represent gene identity and binned expression values are encoded through a separate embedding layer. The two embeddings are then summed to form the final input representation for the Performer encoder.

During pretraining, scBERT adopts a self-supervised masked modeling task similar to BERT but specifically adapted to the characteristics of scRNA-seq data. Given that single-cell expression matrices are highly sparse and contain many zeros caused by technical dropouts, the model randomly masks a proportion of the non-zero expression entries (15%). The masking target is the expression value itself rather than the gene token and The training objective is to reconstruct the masked gene expression value based on the remaining unmasked expressions within the same cell, including zero and non-zero genes. By learning to predict these quantitative values from surrounding gene contexts, scBERT captures biologically meaningful co-expression patterns and long-range gene–gene dependencies.

### Geneformer

Around the same time as scBERT, Geneformer proposed a conceptually different approach to modeling scRNA-seq data. Instead of combining gene identity and expression embeddings, Geneformer directly adopts the `BertModel` architecture from the Transformers library. As a result, it takes only gene tokens as input and does not explicitly incorporate expression values.

To encode quantitative information, Geneformer introduces an innovative rank-based encoding strategy, in which genes within each cell are ranked by their expression levels, and only the top 2,048 most highly expressed genes are taken as input. Instead of relying on absolute values, this approach allows the model to focus on genes that are most informative for defining cellular states. Because the relative order of genes within a cell is consistent, rank-based representation remains robust to technical noise and batch effects compared with raw count–based methods. Additionally, this approach significantly reduces the input sequence length compared to using all genes, improving computational efficiency.

Based this strategy, the authors merged multiple public datasets to construct a large pretraining corpus, Genecorpus-30M, containing approximately 29.9 million human single-cell transcriptomes across a wide range of tissues. This large and diverse dataset enables Geneformer to learn generalizable representations of gene–gene relationships that can be effectively transferred to various downstream biological tasks.

### scGPT

Similar to scBERT, scGPT follows a comparable preprocessing and pretraining strategy. In the preprocessing stage, gene expression values are discretized through a value binning process to mitigate scale differences across genes and cells. During pretraining, scGPT also adopts a masked prediction objective in which expression values are randomly masked and reconstructed from contextual information within the same cell.

For large-scale pretraining, scGPT was trained on over 33 million single-cell transcriptomes collected from more than 1,600 publicly available scRNA-seq datasets sourced from CellxGene. These datasets encompass a wide range of tissues, species, and sequencing technologies, providing a diverse foundation for learning generalizable gene-expression representations.

However, scGPT differs from scBERT in several key aspects. First, the model only takes genes with non-zero expression values as input, thereby focusing on biologically active genes and their co-expression relationships rather than the entire gene space. Second, scGPT introduces learnable gene embeddings instead of relying on static embeddings derived from `gene2vec`, allowing the model to jointly optimize both gene identity and expression representations during training. These design choices enable scGPT to capture richer functional relationships between actively expressed genes and improve generalization across diverse single-cell datasets.

### scFoundation

Compared with previous models, scFoundation introduces a more elaborate pretraining framework that explicitly accounts for variations in sequencing depth. In the data preprocessing stage, expression matrices are log-transformed and normalized in a manner similar to scBERT and scGPT, but the model directly uses continuous expression values instead of discretized bins. This choice allows scFoundation to preserve quantitative information and avoid the information loss that occurs during discretization.

Before normalization, it chose a proportion of all cells and their expression profile is first randomly downsampled to simulate low read-depth conditions, while the original full-depth profile is retained for comparison. Both the downsampled and original data are then log-transformed and normalized, and their total read counts before and after downsampling are concatenated at the end of the sequence as additional scalar features to form the final input representation.

In pretraining, unlike scBERT and scGPT, which mask and predict discretized values, scFoundation performs regression directly on continuous expression values. It employs two learning objectives based on the input type:

1. When the input is the original unsampled expression vector, the model learns gene–gene dependencies within a single cell.
2. When the input is the downsampled variant, the model captures relationships across cells with different sequencing depths.

In terms of architecture, scFoundation follows an asymmetric encoder–decoder Transformer design. The encoder receives only the non-zero expressed genes as input and produces a compact latent representation that captures the relationships among actively expressed genes. This latent representation can also be used as a cell embedding for downstream tasks such as cell type annotation.

In contrast, the decoder operates on the complete gene sequence, including both zero- and non-zero-expressed genes. After embedding the full sequence, the decoder replaces the positions corresponding to non-zero genes with the latent representations produced by the encoder.

This design allows the decoder to reconstruct the full expression profile while integrating contextual information from both observed and unobserved genes to achieve comprehensive generative representation of single-cell transcriptomes.

## Finetuning on Downstream Tasks

As discussed above, this study mainly focuses on three representative downstream tasks in single-cell analysis: cell type annotation, perturbation prediction, and gene regulatory network (GRN) inference. These tasks serve as standard benchmarks for evaluating the transferability and biological interpretability of pretrained single-cell foundation models. Although scBERT, Geneformer, scGPT, and scFoundation share similar Transformer-based architectures, they differ substantially in how they are fine-tuned or applied to these downstream tasks.

The coverage of these downstream tasks across the four baseline models is summarized in the table below:

| Model | Cell Type Annotation | Perturbation Prediction | GRN Inference |
|-------|---------------------|------------------------|---------------|
| scBERT | ✓ | --- | --- |
| Geneformer | ✓ | ~* | --- |
| scGPT | ✓ | ✓ | ✓ |
| scFoundation | ✓ | ✓ | ✓ |

**Notes:** ✓ indicates that the model was formally evaluated on the task; ~ denotes that a related or similar analysis was performed; "---" indicates that the task was not evaluated.

*\* The "similar analysis" for Geneformer refers to in silico perturbation, which simulates gene perturbations within the pretrained model rather than performing regression-based prediction on real Perturb-seq datasets.*

### Cell Type Annotation

Cell type annotation is the most common downstream task for evaluating pretrained single-cell foundation models, as it measures how well the learned representations can be transferred to supervised classification settings. All the baseline models have been fine-tuned on this task.

Despite differences in pretraining and architecture, these models follow a common fine-tuning paradigm. They first obtain token-level representations from the pretrained encoder and then derive a cell-level embedding via an aggregation step (e.g., a **CLS** token, mean/weighted pooling, or other learned aggregation). The derived cell embedding is then used as the input to a lightweight classifier, which outputs the predicted cell type for each cell.

During fine-tuning, the model is trained with the provided cell-type labels using a cross-entropy loss function, either updating only the classifier with the weight of the encoder frozen or optimizing the entire model end-to-end.

For evaluation, cell type annotation is formulated as a multi-class classification task, and the performance is typically measured using standard classification metrics. Common evaluation metrics include precision, recall, accuracy, and macro F1-score. Precision measures the proportion of correctly predicted cells among all cells predicted for a given cell type, while recall measures the proportion of correctly predicted cells among all cells that truly belong to a given cell type. Accuracy represents the overall proportion of correctly classified cells across all cell types. Macro F1-score computes the F1-score for each cell type separately and then takes the unweighted mean across all cell types, making it particularly useful for evaluating performance on imbalanced datasets where some cell types may have significantly fewer samples than others. These metrics provide complementary perspectives on model performance and help identify potential bias towards majority cell types.

### Perturbation Prediction

Among the four models, scGPT and scFoundation have been explicitly evaluated on the perturbation prediction task, which aims to predict gene expression changes under genetic perturbations. scBERT did not include this task in its evaluation, while Geneformer proposed a related but conceptually different analysis termed *in silico perturbation*. Unlike perturbation prediction models that are trained and tested on real perturb-seq datasets, Geneformer's *in silico perturbation* does not rely on experimental perturbation data. Instead, it simulates the deletion or activation of specific genes directly within the pretrained model by manipulating the rank-based gene tokens in the input sequence. Instead, it simulates gene perturbations within the pretrained model by adjusting the rank positions of gene tokens in the input and observe how virtual gene deletions or activations change the resulting cell representations.

To represent perturbation information, scGPT introduces a separate embedding layer that encodes the perturbed genes. This perturbation embedding is then added to the corresponding gene embedding and expression value embedding, forming a combined representation for each gene. The Transformer encoder takes the sequence of combined embeddings as input to capture gene–gene and gene–perturbation interactions within the same contextual space. The encoder output is subsequently passed to a regression head that predicts continuous gene expressions corresponding to the perturbed cell state. The model is trained using a mean squared error loss between the predicted and actual expression values from perturb-seq experiments.

scGPT evaluated the results of perturbation prediction task using the *Pearson-δ* metric, which quantifies the correlation between the predicted and observed gene expression changes after perturbation. In addition, the authors report *Pearson-δ DE* (on differentially expressed genes), computed on the top 20 most significantly perturbed genes for each experiment. These DE genes were identified by ranking genes according to the magnitude of their observed expression change between perturbed and control cells. Evaluating on this subset provides a more biologically meaningful measure of how accurately the model captures key regulatory responses rather than minor background fluctuations.

scGPT was benchmarked against two baseline models:

1. GEARS, a graph neural network–based method for perturbation prediction.
2. A linear regression model as a simple statistical baseline.

In all comparisons, scGPT consistently achieved the highest correlation scores, outperforming previous methods by a margin of 5–20%. This demonstrates the model's superior ability to generalize to unseen perturbations and accurately predict post-perturbation gene expression profiles.

scFoundation, on the other hand, was integrated with the Gears model to accomplish perturbation prediction. The pretrained scFoundation model was used as a feature extractor to generate gene embeddings from both control and perturbed data. These embeddings were then fed into the Gears architecture, which models gene regulatory relationships using a graph neural network. By combining the rich representations learned by scFoundation with Gears' ability to capture complex gene–gene interactions, the integrated model aimed to improve perturbation prediction accuracy.

For evaluation, scFoundation employed the same *Pearson-δ* and *Pearson-δ on DE genes* metrics as scGPT. The results showed that the integrated model outperformed Gears alone, demonstrating the benefit of leveraging pretrained single-cell representations for downstream perturbation tasks.

### Gene Regulatory Network Inference

scGPT adopts two strategies to uncover gene regulatory networks. The first approach constructs a data-independent regulatory network based on the pretrained gene embeddings. After large-scale pretraining, each gene in the model vocabulary is represented by a embedding vector that encodes semantic information about gene function and co-expression learned from millions of single-cell transcriptomes. Pairwise cosine similarity between gene embeddings is computed to build a global gene similarity network, followed by Louvain clustering to identify coherent gene modules, or gene programs. Modules containing at least five genes are retained, and pathway enrichment analyses using Reactome or Gene Ontology (GO) databases are conducted to validate their biological relevance. This embedding-based strategy provides a zero-shot capability to perform GRN inference across datasets without task-specific fine-tuning.

The second approach extracts cell-state-specific regulatory interactions directly from the Transformer's multi-head self-attention mechanism. After fine-tuning scGPT on a target dataset, attention weights from the final Transformer layer are interpreted as context-dependent dependencies between genes, where each attention score reflects the influence of one gene on another within a given cellular state. By comparing attention matrices between different biological conditions, the model identifies genes exhibiting the largest attention differences, corresponding to potential transcription factor (TF) target interactions. These differential attention patterns have been validated against ChIP-Atlas and curated pathway databases, confirming that scGPT can recover known TF–target relationships and capture condition-specific regulatory cascades.

Similarly to the first strategy of scGPT, scFoundation also infers gene regulatory relationships based on pretrained gene embeddings. It computes cosine similarity between embedding derived from the pretrained model to construct a similarity network, followed by Leiden clustering analysis to identify functional gene modules and performing enrichment analysis to validate their biological relevance. However, unlike scGPT, scFoundation further incorporates the SCENIC framework to identify transcription factors and their cell-type-specific regulatory targets. This additional step enables more biologically interpretable and cell-specific GRN inference.
