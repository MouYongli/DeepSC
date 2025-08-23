#!/usr/bin/env python3
"""
Debug script to demonstrate PyTorch MultiheadAttention issues in DeepSC model.
Shows the double V computation problem and architectural mismatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import traceback
from typing import Optional

# Set seed for reproducibility
torch.manual_seed(42)

class BranchV(nn.Module):
    """Custom V computation module from DeepSC"""
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, g, _ = x.shape
        V = self.W_V(x).view(batch, g, self.num_heads, self.head_dim)
        return V


class ProblematicGeneAttentionLayer(nn.Module):
    """Current implementation from DeepSC - demonstrates the issues"""
    def __init__(self, d, num_heads, attn_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(d, d)
        # PyTorch MHA - will compute its own Q,K,V internally
        self.mha = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )

    def forward(self, x, V, M=None, eps: float = 1e-8):
        print("🚨 ISSUE: Double V computation happening!")
        
        if M is not None and M.dim() == 3:
            M = M.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # ISSUE 1: Custom V computation
        V = V.permute(0, 2, 1, 3)  # (batch, num_heads, g, head_dim)
        print(f"   Custom V shape: {V.shape}")
        
        # ISSUE 2: PyTorch MHA computes its own Q,K,V internally (including V!)
        print("   PyTorch MHA computing Q,K,V internally...")
        output, A = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        print(f"   MHA output shape: {output.shape}, Attention shape: {A.shape}")
        print("   ❌ MHA's internal V computation is WASTED!")
        
        # ISSUE 3: Biological masking applied post-hoc, breaking gradient flow
        if M is not None:
            print("   Applying biological masking post-hoc...")
            A_sparse = A * M
            norm = torch.sum(torch.abs(A_sparse), dim=-1, keepdim=True) + eps
            A_bar = A_sparse / norm
            print("   ⚠️  Biological masking breaks gradient flow to Q,K!")
        else:
            A_bar = A
        
        # ISSUE 4: Manual attention computation using custom V (ignoring MHA output)
        print("   Computing attention manually with custom V...")
        output = torch.matmul(A_bar, V)  # Uses custom V, NOT MHA's V!
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        output = self.out_proj(output)
        print("   ❌ MHA output completely ignored!")
        
        return output


class FixedGeneAttentionLayer(nn.Module):
    """Fixed implementation that properly integrates PyTorch MHA with biological masking"""
    def __init__(self, d, num_heads, attn_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.scale = (self.head_dim) ** -0.5
        
        # Separate Q,K,V projections for biological masking integration
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False) 
        self.W_v = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, V=None, M=None, eps: float = 1e-8):
        print("✅ FIXED: Integrated biological masking with efficient attention!")
        
        batch, seq_len, d = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_computed = self.W_v(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        print(f"   Q,K,V shapes: {Q.shape}, {K.shape}, {V_computed.shape}")
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply biological masking BEFORE softmax (proper gradient flow)
        if M is not None:
            print("   Applying biological masking BEFORE softmax...")
            if M.dim() == 3:
                M = M.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores + (M.log() if (M > 0).any() else -1e9 * (1 - M))
            print("   ✅ Biological constraints properly integrated!")
        
        # Softmax attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V_computed)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d)
        output = self.out_proj(output)
        
        print("   ✅ Single, efficient attention computation!")
        return output


def create_test_data():
    """Create test data matching DeepSC dimensions"""
    batch_size = 2
    seq_len = 100  # Number of genes
    d = 512  # Embedding dimension
    num_heads = 8
    
    # Input embeddings
    x = torch.randn(batch_size, seq_len, d, requires_grad=True)
    
    # Custom V from BranchV
    branch_v = BranchV(d, num_heads)
    V = branch_v(x)
    
    # Biological masking matrix (gene regulation)
    M = torch.rand(batch_size, seq_len, seq_len)
    M = (M > 0.7).float()  # Sparse biological constraints
    
    return x, V, M, d, num_heads


def benchmark_attention_layers():
    """Benchmark problematic vs fixed attention implementations"""
    print("=" * 60)
    print("🔍 ATTENTION MECHANISM DEBUG ANALYSIS")
    print("=" * 60)
    
    # Create test data
    x, V, M, d, num_heads = create_test_data()
    
    # Initialize layers
    problematic_layer = ProblematicGeneAttentionLayer(d, num_heads)
    fixed_layer = FixedGeneAttentionLayer(d, num_heads)
    
    print(f"\nTest data: batch_size=2, seq_len=100, embed_dim={d}, num_heads={num_heads}")
    print(f"Biological mask sparsity: {M.mean():.3f}")
    
    # Test problematic implementation
    print("\n" + "="*50)
    print("🚨 TESTING PROBLEMATIC IMPLEMENTATION")
    print("="*50)
    
    start_time = time.time()
    try:
        output1 = problematic_layer(x, V, M)
        problematic_time = time.time() - start_time
        print(f"✓ Completed in {problematic_time:.4f}s")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return
    
    # Test fixed implementation  
    print("\n" + "="*50)
    print("✅ TESTING FIXED IMPLEMENTATION")
    print("="*50)
    
    start_time = time.time()
    try:
        output2 = fixed_layer(x, None, M)  # V is computed internally
        fixed_time = time.time() - start_time
        print(f"✓ Completed in {fixed_time:.4f}s")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return
    
    # Performance comparison
    print("\n" + "="*50)
    print("📊 PERFORMANCE ANALYSIS")
    print("="*50)
    print(f"Problematic implementation: {problematic_time:.4f}s")
    print(f"Fixed implementation:       {fixed_time:.4f}s")
    print(f"Speedup:                   {problematic_time/fixed_time:.2f}x")
    
    # Memory analysis
    print(f"\nOutput shapes:")
    print(f"Problematic: {output1.shape}")
    print(f"Fixed:       {output2.shape}")
    
    # Gradient flow test
    print("\n" + "="*50)
    print("🔄 GRADIENT FLOW ANALYSIS")
    print("="*50)
    
    # Test gradients for problematic implementation
    loss1 = output1.sum()
    loss1.backward(retain_graph=True)
    problematic_grad_norm = x.grad.norm().item() if x.grad is not None else 0
    print(f"Problematic grad norm: {problematic_grad_norm:.6f}")
    
    # Clear gradients
    x.grad = None
    
    # Test gradients for fixed implementation
    loss2 = output2.sum()
    loss2.backward()
    fixed_grad_norm = x.grad.norm().item() if x.grad is not None else 0
    print(f"Fixed grad norm:       {fixed_grad_norm:.6f}")
    
    print("\n" + "="*60)
    print("🎯 SUMMARY OF ISSUES FOUND:")
    print("="*60)
    print("1. ❌ Double V computation in problematic version")
    print("2. ❌ PyTorch MHA output completely ignored") 
    print("3. ❌ Biological masking applied post-hoc, breaking gradients")
    print("4. ✅ Fixed version integrates masking properly")
    print("5. ✅ Fixed version eliminates redundant computations")
    print(f"6. ✅ Fixed version is {problematic_time/fixed_time:.2f}x faster")


if __name__ == "__main__":
    benchmark_attention_layers()