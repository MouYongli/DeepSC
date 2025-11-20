# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited
# Fine-tuning script for cell type annotation with scFoundation

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from load import load_model_frommmf, gatherData, seed_all
from celltype_dataset import create_data_splits


class CellTypeClassifier(nn.Module):
    """
    Cell type classification model based on scFoundation
    """

    def __init__(self, ckpt_path, num_classes, finetune_layers=2, finetune_all_encoder=False,
                 freeze_embeddings=True, dropout=0.1):
        """
        Args:
            ckpt_path: Path to scFoundation checkpoint
            num_classes: Number of cell types to classify
            finetune_layers: Number of encoder layers to fine-tune (from the end), ignored if finetune_all_encoder=True
            finetune_all_encoder: If True, fine-tune all encoder layers; if False, only fine-tune last N layers
            freeze_embeddings: Whether to freeze token and position embeddings
            dropout: Dropout rate for classifier head
        """
        super().__init__()
        self.ckpt_path = ckpt_path
        self.num_classes = num_classes
        self.finetune_layers = finetune_layers
        self.finetune_all_encoder = finetune_all_encoder
        self.freeze_embeddings = freeze_embeddings

        # Load pre-trained model
        print(f"Loading pre-trained model from {ckpt_path}")
        model, model_config = load_model_frommmf(ckpt_path, key='cell')

        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        self.model_config = model_config

        hidden_dim = model_config['encoder']['hidden_dim']
        num_encoder_layers = len(self.encoder.transformer_encoder)
        print(f"Model config: hidden_dim={hidden_dim}, encoder_layers={num_encoder_layers}")

        # Freeze embeddings if specified
        if freeze_embeddings:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print("✓ Froze token and position embeddings")
        else:
            print("✓ Token and position embeddings will be fine-tuned")

        # Handle encoder fine-tuning strategy
        if finetune_all_encoder:
            # Fine-tune all encoder layers
            print(f"✓ Fine-tuning ALL {num_encoder_layers} encoder layers")
            for _, param in self.encoder.named_parameters():
                param.requires_grad = True
        else:
            # Freeze all encoder layers first
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False

            # Unfreeze the last N encoder layers for fine-tuning
            if finetune_layers > 0:
                layers_to_finetune = list(range(num_encoder_layers - finetune_layers, num_encoder_layers))
                print(f"✓ Fine-tuning last {finetune_layers} encoder layers: {layers_to_finetune}")

                for layer_idx in layers_to_finetune:
                    for _, param in self.encoder.transformer_encoder[layer_idx].named_parameters():
                        param.requires_grad = True
            else:
                print("✓ All encoder layers frozen (feature extraction mode)")

        # Classification head
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-6)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Count trainable parameters
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def forward(self, x):
        """
        Args:
            x: (batch_size, 19264) gene expression tensor

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Identify non-zero genes (value mask)
        value_labels = x > 0  # (B, 19264)

        # Gather only non-zero genes to save computation
        x_gathered, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])

        # Create position IDs for genes
        data_gene_ids = torch.arange(19264, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.model_config['pad_token_id'])

        # Token embedding (AutoBin discretization)
        x_emb = self.token_emb(x_gathered.unsqueeze(2).float(), output_weight=0)

        # Add position embedding
        pos_emb = self.pos_emb(position_gene_ids)
        x_emb = x_emb + pos_emb

        # Encoder
        encoder_output = self.encoder(x_emb, x_padding)  # (B, L, hidden_dim)

        # Pool to cell-level embedding (max pooling)
        cell_emb, _ = torch.max(encoder_output, dim=1)  # (B, hidden_dim)

        # Normalize
        cell_emb = self.norm(cell_emb)

        # Classifier
        logits = self.classifier(cell_emb)  # (B, num_classes)

        return logits


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, gradient_accumulation_steps=1):
    """Train for one epoch

    Args:
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    optimizer.zero_grad()  # Initialize gradients at the start

    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)
        labels = batch['label'].to(device)

        # Forward
        logits = model(x)
        loss = criterion(logits, labels)

        # Normalize loss by accumulation steps
        loss = loss / gradient_accumulation_steps

        # Backward
        loss.backward()

        # Update parameters every N steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        # Metrics (use unnormalized loss for display)
        total_loss += loss.item() * gradient_accumulation_steps
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'lr': f'{current_lr:.2e}'
        })

    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch=None):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    pbar = tqdm(val_loader, desc=desc)

    for batch in pbar:
        x = batch['x'].to(device)
        labels = batch['label'].to(device)

        # Forward
        logits = model(x)
        loss = criterion(logits, labels)

        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def test_model(model, test_loader, device, label_mapping, save_dir):
    """Test the model and generate detailed report"""
    model.eval()
    all_preds = []
    all_labels = []
    all_label_strs = []

    print("\n" + "="*60)
    print("Testing model...")
    print("="*60)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch['x'].to(device)
            labels = batch['label']
            label_strs = batch['label_str']

            # Forward
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_label_strs.extend(label_strs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision (macro):    {precision_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (macro):       {recall_macro:.4f}")
    print(f"Recall (weighted):    {recall_weighted:.4f}")
    print(f"F1 (macro):           {f1_macro:.4f}")
    print(f"F1 (weighted):        {f1_weighted:.4f}")
    print(f"{'='*60}\n")

    # Classification report
    target_names = [str(label_mapping[i]) for i in range(len(label_mapping))]
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print("Classification Report:")
    print(report)

    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'precision_weighted': float(precision_weighted),
        'recall_macro': float(recall_macro),
        'recall_weighted': float(recall_weighted),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'classification_report': report
    }

    results_path = os.path.join(save_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(save_dir, 'confusion_matrix.npy')
    np.save(cm_path, cm)
    print(f"Confusion matrix saved to {cm_path}")

    return accuracy, precision_macro, recall_macro, f1_macro


def main(args):
    # Set random seed
    seed_all(args.seed)

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"celltype_finetune_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nOutput directory: {save_dir}\n")

    # Save args
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    train_loader, val_loader, test_loader, num_classes, label_mapping = create_data_splits(
        h5ad_path=args.data,
        label_key=args.label_key,
        gene_list_path=args.gene_list,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_key=args.split_key,
        random_seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Save label mapping
    with open(os.path.join(save_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=2)

    # Create model
    print(f"\n{'='*60}")
    print("Creating model...")
    print(f"{'='*60}\n")

    model = CellTypeClassifier(
        ckpt_path=args.checkpoint,
        num_classes=num_classes,
        finetune_layers=args.finetune_layers,
        finetune_all_encoder=args.finetune_all_encoder,
        freeze_embeddings=not args.no_freeze_embeddings,  # Invert the flag
        dropout=args.dropout
    )
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer - use different learning rates for different parts
    params_to_optimize = [
        {'params': [p for n, p in model.encoder.named_parameters() if p.requires_grad],
         'lr': args.lr * 0.1},  # Lower LR for fine-tuned encoder layers
        {'params': model.classifier.parameters(),
         'lr': args.lr}  # Higher LR for classifier head
    ]

    optimizer = AdamW(params_to_optimize, weight_decay=args.weight_decay)

    # Learning rate scheduler (optional)
    scheduler = None
    if args.use_scheduler:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs]
        )
        print(f"Using learning rate scheduler: Warmup ({args.warmup_epochs} epochs) + Cosine Annealing")
    else:
        print(f"Using constant learning rate: {args.lr:.2e}")

    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size}")
    print(f"{'='*60}\n")

    best_val_f1 = 0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = validate(
            model, val_loader, criterion, device, epoch
        )

        # Update scheduler (if using)
        if scheduler is not None:
            scheduler.step()

        # Log
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")
        # Show learning rates for both parameter groups
        if len(optimizer.param_groups) > 1:
            print(f"  LR (encoder): {optimizer.param_groups[0]['lr']:.2e}, LR (classifier): {optimizer.param_groups[1]['lr']:.2e}")
        else:
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'num_classes': num_classes,
                'label_mapping': label_mapping,
            }, best_model_path)
            print(f"  ✓ Saved best model (F1: {val_f1:.4f})")

        print()

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': num_classes,
        'label_mapping': label_mapping,
    }, final_model_path)

    # Save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*60}\n")

    # Test with best model
    print("Loading best model for testing...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_acc, test_precision, test_recall, test_f1 = test_model(
        model, test_loader, device, label_mapping, save_dir
    )

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune scFoundation for cell type annotation')

    # Data
    parser.add_argument('--data', type=str, required=True,
                       help='Path to h5ad file')
    parser.add_argument('--label_key', type=str, default='cell_type',
                       help='Key in adata.obs for cell type labels')
    parser.add_argument('--gene_list', type=str, default='./OS_scRNA_gene_index.19264.tsv',
                       help='Path to gene vocabulary file')
    parser.add_argument('--split_key', type=str, default=None,
                       help='Key in adata.obs for existing train/val/test split')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Train split ratio (if creating new split)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test split ratio')

    # Model
    parser.add_argument('--checkpoint', type=str, default='./models/models.ckpt',
                       help='Path to scFoundation checkpoint')
    parser.add_argument('--finetune_layers', type=int, default=2,
                       help='Number of encoder layers to fine-tune (from the end, ignored if --finetune_all_encoder)')
    parser.add_argument('--finetune_all_encoder', action='store_true', default=False,
                       help='Fine-tune all encoder layers instead of just last N layers')
    parser.add_argument('--no_freeze_embeddings', action='store_true', default=False,
                       help='Do NOT freeze embeddings (allow fine-tuning token and position embeddings)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for classifier')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps to accumulate gradients before updating (default: 1, no accumulation)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for classifier (encoder uses 0.1x)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--use_scheduler', action='store_true', default=False,
                       help='Use learning rate scheduler (warmup + cosine annealing). Default: constant LR')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                       help='Number of warmup epochs (only used if --use_scheduler is set)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Other
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    main(args)
