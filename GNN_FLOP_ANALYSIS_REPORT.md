# GNN FLOP Analysis Report: SGFN vs ESGNN

## Executive Summary

This report provides a comprehensive analysis of the computational complexity (FLOPs) for the Graph Neural Network (GNN) components in SGFN and ESGNN models. The analysis reveals that **ESGNN is 1.30x more efficient than SGFN** in terms of GNN FLOPs.

## Key Findings

- **SGFN**: 184,832,000 FLOPs (2 FAN layers)
- **ESGNN**: 141,644,800 FLOPs (1 EGNN layer with FAN + E_GCL)
- **Efficiency Ratio**: ESGNN/SGFN = 0.766 (ESGNN is more efficient)

## Mathematical Formulas

### 1. MLP FLOPs Formula

For an MLP with layers `[input_dim, hidden_1, hidden_2, ..., output_dim]`:

```
FLOPs = input_dim × hidden_1 + hidden_1 × hidden_2 + ... + hidden_n × output_dim
```

This counts matrix multiplications where each element requires `input_dim` multiplications.

### 2. FAN Layer Components

#### a) Triplet MLP
- **Input**: `[x_i, edge_feature, x_j] = [256, 256, 256] = 768`
- **Hidden**: `[256 + 256] = 512`
- **Output**: `256`
- **Formula**: `num_edges × (768 × 512 + 512 × 256) = num_edges × 524,288`

#### b) Q, K, V Projections
- **Q_proj**: `dim_node → dim_node (256 → 256)`
- **K_proj**: `dim_edge → dim_edge (256 → 256)`
- **V_proj**: `dim_node → dim_atten (256 → 256)`
- **Formula**: `num_edges × (256 × 256) × 3 = num_edges × 196,608`

#### c) Attention MLP
- **Input**: `[x_i_proj, edge_proj] = [32, 32] = 64` (per head)
- **Hidden**: `[32 + 32] = 64`
- **Output**: `32` (per head)
- **Formula**: `num_edges × (64 × 64 + 64 × 32) = num_edges × 6,144`

#### d) Softmax
- **Formula**: `num_edges × dim_edge_proj × num_heads = num_edges × 32 × 8`

#### e) Element-wise Multiplication
- **Formula**: `num_edges × dim_atten = num_edges × 256`

#### f) Node Update MLP
- **Input**: `[x_ori, aggregated_value] = [256, 256] = 512`
- **Hidden**: `[256 + 256] = 512`
- **Output**: `256`
- **Formula**: `num_nodes × (512 × 512 + 512 × 256) = num_nodes × 393,216`

### 3. EGNN Layer Components

#### a) FAN Component
Same as FAN layer components above.

#### b) E_GCL Edge MLP
- **Input**: `[source, target, radial, edge_attr] = [256, 256, 1, 256] = 769`
- **Hidden**: `256`
- **Output**: `256`
- **Formula**: `num_edges × (769 × 256 + 256 × 256) = num_edges × 262,400`

#### c) E_GCL Attention MLP
- **Input**: `256`
- **Output**: `1`
- **Formula**: `num_edges × 256`

#### d) E_GCL Node MLP
- **Input**: `[hidden_nf + input_nf] = [256 + 256] = 512`
- **Hidden**: `256`
- **Output**: `256`
- **Formula**: `num_nodes × (512 × 256 + 256 × 256) = num_nodes × 196,608`

#### e) E_GCL Coordinate MLP
- **Input**: `256`
- **Hidden**: `256`
- **Output**: `1`
- **Formula**: `num_edges × (256 × 256 + 256 × 1) = num_edges × 65,792`

#### f) Embedding Layers
- **embedding_in**: `256 → 256`
- **embedding_out**: `256 → 256`
- **Formula**: `num_nodes × (256 × 256) × 2 = num_nodes × 131,072`

## Detailed FLOP Breakdown

### SGFN (2 FAN layers)
```
Single FAN layer: 92,416,000 FLOPs
├── Triplet MLP: 52,428,800 (56.7%)
├── Q Projection: 6,553,600 (7.1%)
├── K Projection: 6,553,600 (7.1%)
├── V Projection: 6,553,600 (7.1%)
├── Attention MLP: 614,400 (0.7%)
├── Softmax: 25,600 (0.0%)
├── Element-wise Mult: 25,600 (0.0%)
└── Node Update: 19,660,800 (21.3%)

Total SGFN: 184,832,000 FLOPs
```

### ESGNN (1 EGNN layer with FAN)
```
Single EGNN layer: 141,644,800 FLOPs
├── FAN Component: 92,416,000 (65.2%)
│   ├── Triplet MLP: 52,428,800
│   ├── Q,K,V Projections: 19,660,800
│   ├── Attention MLP: 614,400
│   ├── Softmax: 25,600
│   ├── Element-wise Mult: 25,600
│   └── Node Update: 19,660,800
├── E_GCL Component: 49,228,800 (34.8%)
│   ├── Edge MLP: 26,240,000 (18.5%)
│   ├── Attention MLP: 25,600 (0.0%)
│   ├── Node MLP: 9,830,400 (6.9%)
│   ├── Coord MLP: 6,579,200 (4.6%)
│   ├── Embedding In: 3,276,800 (2.3%)
│   └── Embedding Out: 3,276,800 (2.3%)

Total ESGNN: 141,644,800 FLOPs
```

## Model Configurations

### SGFN Configuration
- **Method**: `fan`
- **Layers**: `2`
- **Heads**: `8`
- **Node dim**: `256`
- **Edge dim**: `256`
- **Hidden dim**: `256`

### ESGNN Configuration
- **Method**: `esgnn`
- **Layers**: `1`
- **Heads**: `8`
- **Node dim**: `256`
- **Edge dim**: `256`
- **Hidden dim**: `256`
- **With FAN**: `true`
- **With X**: `false`

## Analysis Parameters

- **Graph size**: 50 nodes, 100 edges
- **Feature dimensions**: 256 for both node and edge features
- **Attention heads**: 8
- **Temperature scaling**: √(dim_edge_proj) = √32 ≈ 5.66

## Verification

The calculations have been verified through:
1. **Manual computation** of key components
2. **Cross-validation** with function implementations
3. **Step-by-step breakdown** of each operation

All calculations match between manual and automated methods.

## Conclusions

1. **ESGNN is more efficient** than SGFN by a factor of 1.30x
2. **The efficiency comes from** using a single complex layer instead of multiple simpler layers
3. **FAN component dominates** both architectures (65.2% of ESGNN FLOPs)
4. **E_GCL adds significant complexity** but is still more efficient than running FAN twice
5. **The analysis is accurate** and based on actual model implementations

## Recommendations

1. **Use ESGNN** for better computational efficiency
2. **Consider the trade-off** between model complexity and performance
3. **Monitor memory usage** as ESGNN may have different memory patterns
4. **Profile actual runtime** to validate theoretical FLOP analysis

---

*This analysis is based on the actual model implementations in the 3DSSG codebase and verified through multiple calculation methods.*