# ADMF-Trader Complete Strategy Development Pipeline with Robustness Framework

## Overview: Enhanced Three-Phase Scientific Approach

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                  COMPLETE STRATEGY DEVELOPMENT PIPELINE                        │
│                         WITH ROBUSTNESS FRAMEWORK                              │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   PHASE 1:      │     │   PHASE 2:      │     │   PHASE 3:      │       │
│  │ DECOMPOSITION & │ ──▶ │   ANALYSIS &    │ ──▶ │ RECOMPOSITION &  │       │
│  │    MINING       │     │   EXPLORATION   │     │   VALIDATION     │       │
│  │ (Training Data) │     │ (Training Data) │     │  (Test Data)     │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                               │
│  Enhanced with:                                                               │
│  • Parameter Neighborhood Analysis                                            │
│  • Return Distribution Clustering                                             │
│  • Walk-Forward Validation                                                   │
│  • Robustness-Based Selection                                                │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Enhanced Decomposition & Mining with Robustness

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              PHASE 1: DECOMPOSITION & MINING WITH ROBUSTNESS                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                  Walk-Forward Window Management                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │  Window 1   │  │  Window 2   │  │  Window N   │     ...       │    │
│  │  │ Train│Test  │  │ Train│Test  │  │ Train│Test  │               │    │
│  │  │ 2020 │2021  │  │ 20-21│21Q2  │  │ 21-22│22Q2  │               │    │
│  │  └──────┴──────┘  └──────┴──────┘  └──────┴──────┘               │    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
  │                Signal Correlation Analysis                         │    │
  │                                                                     │    │
  │  Correlation Matrix of Strategy Signals:                           │    │
  │  ┌─────────────────────────────────────────────────────────┐     │    │
  │  │         MA_5_20  MA_10_30  RSI_14  BBands  Ensemble     │     │    │
  │  │ MA_5_20   1.00     0.65     0.12    -0.23    0.78      │     │    │
  │  │ MA_10_30  0.65     1.00     0.08    -0.31    0.71      │     │    │
  │  │ RSI_14    0.12     0.08     1.00     0.45    0.54      │     │    │
  │  │ BBands   -0.23    -0.31     0.45     1.00    0.22      │     │    │
  │  │                                                          │     │    │
  │  │ Key Insights:                                           │     │    │
  │  │ • MA strategies highly correlated (reduce allocation)   │     │    │
  │  │ • RSI provides diversification benefit                  │     │    │
  │  │ • BBands offers negative correlation (hedge)            │     │    │
  │  └─────────────────────────────────────────────────────────┘     │    │
  └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │          Enhanced Grid Search with Neighborhood Testing               │  │
│  │                                                                       │  │
│  │  For each parameter combination:                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  1. Test base parameters (e.g., MA(5,20))                   │   │  │
│  │  │  2. Test parameter neighborhood:                            │   │  │
│  │  │     - MA(4,19), MA(4,20), MA(4,21)                        │   │  │
│  │  │     - MA(5,19), [BASE], MA(5,21)                          │   │  │
│  │  │     - MA(6,19), MA(6,20), MA(6,21)                        │   │  │
│  │  │  3. Calculate stability metrics                            │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Shared Indicator Architecture                   │   │  │
│  │  │  (Computes for base + all neighborhood parameters)          │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Enhanced Mining Output                            │  │
│  │  ┌────────────────────────────────────────────────────────────┐     │  │
│  │  │  Per Strategy Storage:                                      │     │  │
│  │  │  • Base parameter performance                               │     │  │
│  │  │  • Neighborhood performance matrix                          │     │  │
│  │  │  • Stability score: f(mean, std, min of neighborhood)      │     │  │
│  │  │  • Walk-forward consistency scores                          │     │  │
│  │  │  • Return distribution characteristics                      │     │  │
│  │  └────────────────────────────────────────────────────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Phase 2: Enhanced Analysis with Robustness Focus

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                PHASE 2: ENHANCED ANALYSIS & EXPLORATION                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Robustness Analysis Tools                        │    │
│  │                                                                     │    │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │    │
│  │  │   Parameter      │  │     Return       │  │  Walk-Forward   │  │    │
│  │  │   Sensitivity    │  │   Distribution   │  │   Consistency   │  │    │
│  │  │    Heatmaps      │  │   Clustering     │  │    Analysis     │  │    │
│  │  └─────────────────┘  └──────────────────┘  └─────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Parameter Stability Visualization                       │    │
│  │                                                                     │    │
│  │  Parameter Surface for MA Strategy:                                │    │
│  │  ┌─────────────────────────────────────────────────────────┐     │    │
│  │  │     Slow MA                                               │     │    │
│  │  │  30 │ . . . . . . . . . . . . . . . .                   │     │    │
│  │  │  25 │ . . . . ▓▓▓▓▓▓▓▓▓▓▓ . . . .    ← Stable Plateau │     │    │
│  │  │  20 │ . . . ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ . . .                     │     │    │
│  │  │  15 │ . . . ▓▓▓▓▓▓▓▓▓▓▓▓▓ . █ . .    ← Fragile Spike  │     │    │
│  │  │  10 │ . . . . . . . . . . . . . . .                     │     │    │
│  │  │     └─────────────────────────────────                   │     │    │
│  │  │       5    10    15    20    25   Fast MA                │     │    │
│  │  └─────────────────────────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Return Distribution Analysis                           │    │
│  │                                                                     │    │
│  │  Clustering strategies by return characteristics:                   │    │
│  │  ┌─────────────────────────────────────────────────────────┐     │    │
│  │  │  Cluster 1: Trend Following                               │     │    │
│  │  │  - High returns in trending markets                       │     │    │
│  │  │  - Stable parameter neighborhoods                         │     │    │
│  │  │  - Members: MA(5,20), MA(10,30), ...                     │     │    │
│  │  │                                                           │     │    │
│  │  │  Cluster 2: Mean Reversion                               │     │    │
│  │  │  - High returns in range-bound markets                   │     │    │
│  │  │  - Sensitive to parameter changes                        │     │    │
│  │  │  - Members: RSI(14), BB(20,2), ...                       │     │    │
│  │  └─────────────────────────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │          Robustness-Enhanced Strategy Selection                     │    │
│  │                                                                     │    │
│  │  Selection Criteria:                                                │    │
│  │  1. Stability Score > threshold (prefer plateaus over spikes)      │    │
│  │  2. Walk-forward consistency (works across multiple windows)       │    │
│  │  3. Complementary return distributions (for ensemble diversity)    │    │
│  │  4. Regime robustness (not overly specialized)                    │    │
│  │                                                                     │    │
│  │  Output: Robust strategy candidates for testing                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Phase 3: Validation with Full Container Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              PHASE 3: VALIDATION WITH ROBUSTNESS TESTING                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐     ┌────────────────────────────────────┐         │
│  │   Test Dataset     │     │  Selected Strategy Configurations   │         │
│  │  (Unseen Data)     │     │    (From Analysis Phase)           │         │
│  └─────────┬──────────┘     └───────────┬────────────────────────┘         │
│            │                             │                                   │
│            └──────────┬──────────────────┘                                  │
│                       ▼                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Test Container Architecture                        │  │
│  │                    (Same as Training Phase)                          │  │
│  │                                                                       │  │
│  │  ┌────────────────┐         ┌─────────────────────────────────┐    │  │
│  │  │  Test Data     │         │      Regime Classifier           │    │  │
│  │  │    Stream      │────────▶│   (Same model as training)      │    │  │
│  │  └────────────────┘         └───────────┬─────────────────────┘    │  │
│  │                                          │                           │  │
│  │                                          ▼                           │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │           Shared Indicator Architecture (Test)              │   │  │
│  │  │                                                             │   │  │
│  │  │  ┌────────────────────────────────────────────────────┐   │   │  │
│  │  │  │       Indicator Hub (Shared Computation)           │   │   │  │
│  │  │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │   │   │  │
│  │  │  │  │MA(5)│ │MA(20)│ │RSI  │ │ATR  │ │BBands│  ...  │   │   │  │
│  │  │  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘        │   │   │  │
│  │  │  └─────┼───────┼───────┼───────┼───────┼────────────┘   │   │  │
│  │  │        └───────┴───────┴───────┴───────┴─────┐          │   │  │
│  │  │                                               ▼          │   │  │
│  │  │  ┌────────────────────────────────────────────────┐    │   │  │
│  │  │  │          Event Bus (Test Instance)             │    │   │  │
│  │  │  └─────┬──────┬──────┬──────┬──────┬──────┬──────┘    │   │  │
│  │  │        ▼      ▼      ▼      ▼      ▼      ▼            │   │  │
│  │  └─────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Test Strategy Containers                        │   │  │
│  │  │         (Isolated State, Shared Indicators)                  │   │  │
│  │  │                                                              │   │  │
│  │  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │   │  │
│  │  │  │   Container 1   │  │   Container 2   │  │ Container N  │  │   │  │
│  │  │  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌──────────┐ │  │   │  │
│  │  │  │ │   Robust   │ │  │ │  Enhanced  │ │  │ │ Ensemble │ │  │   │  │
│  │  │  │ │  MA(5,20)  │ │  │ │ MA + Filter│ │  │ │Strategies│ │  │   │  │
│  │  │  │ ├────────────┤ │  │ ├────────────┤ │  │ ├──────────┤ │  │   │  │
│  │  │  │ │ Portfolio  │ │  │ │ Portfolio  │ │  │ │Portfolio │ │  │   │  │
│  │  │  │ │   State    │ │  │ │   State    │ │  │ │  State   │ │  │   │  │
│  │  │  │ ├────────────┤ │  │ ├────────────┤ │  │ ├──────────┤ │  │   │  │
│  │  │  │ │   Trade    │ │  │ │   Trade    │ │  │ │  Trade   │ │  │   │  │
│  │  │  │ │  History   │ │  │ │  History   │ │  │ │ History  │ │  │   │  │
│  │  │  │ ├────────────┤ │  │ ├────────────┤ │  │ ├──────────┤ │  │   │  │
│  │  │  │ │   Risk     │ │  │ │   Risk     │ │  │ │   Risk   │ │  │   │  │
│  │  │  │ │  Metrics   │ │  │ │  Metrics   │ │  │ │ Metrics  │ │  │   │  │
│  │  │  │ └────────────┘ │  │ └────────────┘ │  │ └──────────┘ │  │   │  │
│  │  │  └────────────────┘  └────────────────┘  └──────────────┘  │   │  │
│  │  │                                                              │   │  │
│  │  │  Additional Test Containers:                                 │   │  │
│  │  │  • Parameter sensitivity tests (MA(4,19), MA(6,21), etc.)   │   │  │
│  │  │  • Regime transition stress tests                           │   │  │
│  │  │  • Walk-forward out-of-sample windows                       │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              Walk-Forward Test Validation                            │  │
│  │                                                                       │  │
│  │  Test Windows:  [Q1 2024] → [Q2 2024] → [Q3 2024] → [Q4 2024]      │  │
│  │                                                                       │  │
│  │  For each window:                                                    │  │
│  │  1. Create fresh containers for all test strategies                  │  │
│  │  2. Run with shared indicators (computed once per window)            │  │
│  │  3. Collect performance metrics                                      │  │
│  │  4. Verify parameter stability holds                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Final Validation Report                          │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  Strategy Performance Comparison:                           │   │  │
│  │  │                                                              │   │  │
│  │  │  MA(5,20) Robust:                                          │   │  │
│  │  │  - Training Sharpe: 1.5 (neighborhood: 1.3-1.6)            │   │  │
│  │  │  - Test Sharpe: 1.4 (neighborhood: 1.2-1.5) ✓              │   │  │
│  │  │  - Stability Score: 0.92                                   │   │  │
│  │  │  - Walk-forward Consistency: 0.87                          │   │  │
│  │  │                                                              │   │  │
│  │  │  Enhanced MA + Trend Filter:                               │   │  │
│  │  │  - Training Sharpe: 1.6                                    │   │  │
│  │  │  - Test Sharpe: 1.5 ✓                                      │   │  │
│  │  │  - Improved downtrend performance                          │   │  │
│  │  │                                                              │   │  │
│  │  │  Adaptive Ensemble:                                        │   │  │
│  │  │  - Training Sharpe: 1.8                                    │   │  │
│  │  │  - Test Sharpe: 1.7 ✓                                      │   │  │
│  │  │  - Best regime adaptation                                  │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Enhanced YAML Configuration with Robustness

```yaml
pipeline:
  name: "robust_strategy_development"
  
  phase_1_mining:
    data: "data/train_2020_2023.parquet"
    
    walk_forward:
      enabled: true
      training_window: 252
      test_window: 63
      step_size: 21
      
    parameter_search:
      base_grid: "configs/parameter_grid.yaml"
      neighborhood_testing:
        enabled: true
        radius: 2  # Test ±2 steps
        
    robustness_metrics:
      - neighborhood_mean
      - neighborhood_std
      - worst_case_neighbor
      - stability_score
      
  phase_2_analysis:
    tools:
      - parameter_sensitivity_heatmap
      - return_distribution_clustering
      - walk_forward_consistency
      - regime_stability_analysis
      
    selection_criteria:
      min_stability_score: 0.7
      max_neighborhood_std: 0.3
      min_walk_forward_consistency: 0.8
      
  phase_3_test:
    data: "data/test_2024.parquet"
    
    validation:
      test_parameter_drift: true
      drift_range: 0.05  # ±5%
      walk_forward_oos: true
      
    final_selection:
      prefer_robust_over_performance: true
      ensemble_weight_by_stability: true
```

## Key Robustness Enhancements

### 1. **Parameter Neighborhood Testing**
- Every parameter point tests its surrounding neighborhood
- Identifies stable plateaus vs fragile spikes
- Minimal additional computation with shared indicators

### 2. **Walk-Forward Integration**
- Built into all phases, not just final validation
- Each window uses container architecture
- Consistency across windows becomes selection criterion

### 3. **Return Distribution Analysis**
- Clusters strategies by return characteristics
- Identifies complementary strategies for ensembles
- Reveals regime dependencies

### 4. **Robustness-First Selection**
- Stability score weighted equally with performance
- Prefers parameter plateaus over isolated peaks
- Validates stability holds out-of-sample

### 5. **Continuous Validation**
- Parameter sensitivity tested in production
- Drift detection and adaptation
- Ongoing robustness monitoring

## Future Conceptual Enhancements

### 1. **Feedback Loops Between Phases**
- Phase 3 results inform Phase 1 parameter space refinement
- Failed patterns automatically excluded from future searches
- Successful discoveries guide new hypothesis generation

### 2. **Meta-Learning Layer**
Track patterns across multiple pipeline runs:
- Which parameter regions consistently generalize well
- Which analysis discoveries prove robust
- Regime definition effectiveness over time

### 3. **Dynamic Regime Discovery**
- Data-driven regime identification (beyond predefined states)
- Hierarchical regimes with sub-regimes
- Regime stability analysis and evolution tracking

### 4. **Component Dependency Graph**
- Track which indicators drive performance in which conditions
- Identify critical vs optional components
- Quantify component interactions (synergy/redundancy)

### 5. **Adaptive Search Granularity**
- Coarse initial parameter sweep
- Fine-grained search in promising regions
- Ultra-fine analysis near stability boundaries

### 6. **Ensemble Construction as Distinct Phase**
Consider: Mining → Analysis → **Ensemble Design** → Validation
- Dedicated phase for correlation analysis
- Optimal weighting scheme discovery
- Regime-specific vs universal ensembles

### 7. **Production Feedback Integration**
- Monitor strategy degradation in real-time
- Automatic re-analysis triggers
- Strategy health dashboard with early warning system

### 8. **Causality Testing Framework**
Beyond correlation to causation:
- Counterfactual analysis ("what if this signal hadn't fired?")
- Temporal causality validation
- Feature importance ranking

### 9. **Strategy Lineage Tracking**
- Version control for strategy evolution
- Audit trail of modifications and reasons
- Rollback capabilities to previous versions

### 10. **Computational Resource Optimization**
Enhanced Coordinator capabilities:
- Predictive resource allocation
- Dynamic batching based on available resources
- Intelligent caching across phases
- Distributed computation coordination

## System Evolution Path

The architecture should evolve from a **pipeline** to a **learning system**:

```yaml
meta_learning_system:
  continuous_improvement:
    - track_what_works:
        - stable parameter regions
        - effective filters
        - successful ensemble methods
    
    - identify_failure_patterns:
        - overfitting indicators
        - regime misclassification
        - parameter fragility
    
    - apply_learnings:
        - bias searches toward successful patterns
        - avoid known failure modes
        - generate new hypotheses automatically
```

This creates a system that becomes progressively better at finding robust trading strategies through experience.

## Phase 1: Decomposition & Mining (Training Data)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DECOMPOSITION & MINING                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐         ┌─────────────────────────────────────────┐    │
│  │ Training Data  │         │         Regime Classifier                │    │
│  │    Stream      │────────▶│  ┌─────────────────────────────────┐   │    │
│  └────────────────┘         │  │ Adds regime metadata to bars:   │   │    │
│                             │  │ - TRENDING_UP                    │   │    │
│                             │  │ - TRENDING_DOWN                  │   │    │
│                             │  │ - RANGE_BOUND                    │   │    │
│                             │  │ - HIGH_VOLATILITY                │   │    │
│                             │  └─────────────────────────────────┘   │    │
│                             └───────────────┬─────────────────────────┘    │
│                                             │                               │
│                                             ▼                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Shared Indicator Architecture                      │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Indicator Hub (Shared Computation)              │   │  │
│  │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │   │  │
│  │  │  │MA(5) │ │MA(10)│ │MA(15)│ │MA(20)│ │RSI(14)│ │RSI(21)│   │   │  │
│  │  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘   │   │  │
│  │  └─────┼────────┼───────┼───────┼───────┼────────┼──────────┘   │  │
│  │        └────────┴───────┴───────┴───────┴────────┴──────┐       │  │
│  │                                                          ▼       │  │
│  │  ┌─────────────────────────────────────────────────────────┐   │  │
│  │  │                 Event Bus (Broadcast)                    │   │  │
│  │  └─────┬───────┬───────┬───────┬───────┬───────┬──────────┘   │  │
│  │        ▼       ▼       ▼       ▼       ▼       ▼              │  │
│  │  ┌─────────────────────────────────────────────────────┐      │  │
│  │  │           Strategy Containers (100s-1000s)           │      │  │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │      │  │
│  │  │  │MA(5,10)│ │MA(5,15)│ │MA(10,20)│ │RSI+MA  │  ...  │      │  │
│  │  │  │+Regime │ │+Regime │ │+Regime │ │+Regime │       │      │  │
│  │  │  └────────┘ └────────┘ └────────┘ └────────┘       │      │  │
│  │  └──────────────────────────────────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Mining Output Storage                        │  │
│  │  • All trades with regime tags                                   │  │
│  │  • Component signals with strengths                              │  │
│  │  • Performance metrics per strategy                              │  │
│  │  • Regime-specific breakdowns                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Phase 2: Analysis & Exploration (Training Data ONLY)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: ANALYSIS & EXPLORATION                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                     Rich Exploratory Analysis                       │    │
│  │                                                                     │    │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │    │
│  │  │ Jupyter         │  │ Interactive      │  │ Hypothesis      │  │    │
│  │  │ Notebooks       │  │ Dashboards       │  │ Testing         │  │    │
│  │  └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘  │    │
│  │           └──────────────┬──────┴───────────────────┬─┘           │    │
│  │                          ▼                           ▼             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Analysis Container Architecture                        │    │
│  │                                                                     │    │
│  │  ┌──────────────────────────────────────────────────────────┐    │    │
│  │  │         Coordinator (Analysis Mode)                        │    │    │
│  │  │  • Spins up containers for exploratory strategies         │    │    │
│  │  │  • Manages shared data access (read-only)                 │    │    │
│  │  │  • Enables rapid hypothesis testing                       │    │    │
│  │  └──────────────────────────────────────────────────────────┘    │    │
│  │                              │                                     │    │
│  │        ┌────────────────────┼────────────────────┐               │    │
│  │        ▼                    ▼                    ▼               │    │
│  │  ┌──────────┐        ┌──────────┐        ┌──────────┐          │    │
│  │  │ Analysis │        │ Analysis │        │ Analysis │          │    │
│  │  │Container │        │Container │        │Container │          │    │
│  │  │    1     │        │    2     │        │    N     │          │    │
│  │  ├──────────┤        ├──────────┤        ├──────────┤          │    │
│  │  │Experiment│        │Experiment│        │Experiment│          │    │
│  │  │- Counter │        │- Regime  │        │- Signal  │          │    │
│  │  │  trend   │        │  specific│        │  filters │          │    │
│  │  │  filter  │        │  rules   │        │          │          │    │
│  │  └──────────┘        └──────────┘        └──────────┘          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Example Analysis Explorations                          │    │
│  │                                                                     │    │
│  │  1. "MA crossovers perform poorly in downtrends"                   │    │
│  │     → Test: Add trend filter to veto counter-trend signals         │    │
│  │                                                                     │    │
│  │  2. "RSI signals unreliable in high volatility"                    │    │
│  │     → Test: Reduce RSI weight when volatility > threshold          │    │
│  │                                                                     │    │
│  │  3. "Best parameters vary by time of day"                          │    │
│  │     → Test: Time-based parameter switching                          │    │
│  │                                                                     │    │
│  │  All experiments run on TRAINING DATA ONLY to avoid data snooping  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Analysis Output                                  │    │
│  │  • Refined strategy compositions                                   │    │
│  │  • New rules and filters discovered                                │    │
│  │  • Hypothesis test results                                         │    │
│  │  • Candidate strategies for test phase                             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Phase 3: Recomposition & Validation (Test Data)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 3: RECOMPOSITION & VALIDATION                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐     ┌────────────────────────────────────┐         │
│  │   Test Dataset     │     │  Selected Strategy Configurations   │         │
│  │  (Unseen Data)     │     │  (From Analysis Phase)             │         │
│  └─────────┬──────────┘     └───────────┬────────────────────────┘         │
│            │                             │                                   │
│            └──────────┬──────────────────┘                                  │
│                       ▼                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Test Container Architecture                        │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │                  Coordinator (Test Mode)                     │   │  │
│  │  │  • Manages parallel test execution                           │   │  │
│  │  │  • Ensures identical market conditions for all strategies    │   │  │
│  │  │  • Maintains strict data isolation                           │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │           Shared Test Infrastructure                         │   │  │
│  │  │  ┌────────────────┐  ┌────────────────┐                    │   │  │
│  │  │  │ Test Data Feed │  │ Regime         │                    │   │  │
│  │  │  │ (Synchronized) │  │ Classifier     │                    │   │  │
│  │  │  └───────┬────────┘  └───────┬────────┘                    │   │  │
│  │  │          └──────┬─────────────┘                             │   │  │
│  │  │                 ▼                                            │   │  │
│  │  │  ┌─────────────────────────────────────────────────────┐   │   │  │
│  │  │  │        Test Indicator Hub (Shared)                  │   │   │  │
│  │  │  │  Computes all required indicators once              │   │   │  │
│  │  │  └─────────────────────┬───────────────────────────────┘   │   │  │
│  │  └─────────────────────────┼───────────────────────────────────┘   │  │
│  │                            ▼                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Test Strategy Containers                        │   │  │
│  │  │                                                              │   │  │
│  │  │  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐  │   │  │
│  │  │  │   Strategy 1    │ │   Strategy 2    │ │  Strategy N   │  │   │  │
│  │  │  │  ┌───────────┐  │ │  ┌───────────┐  │ │ ┌───────────┐ │  │   │  │
│  │  │  │  │ Original  │  │ │  │ Original  │  │ │ │Discovered │ │  │   │  │
│  │  │  │  │ Best from │  │ │  │ + Counter │  │ │ │in Analysis│ │  │   │  │
│  │  │  │  │ Phase 1   │  │ │  │ Trend     │  │ │ │Phase      │ │  │   │  │
│  │  │  │  │           │  │ │  │ Filter    │  │ │ │           │ │  │   │  │
│  │  │  │  ├───────────┤  │ │  ├───────────┤  │ │ ├───────────┤ │  │   │  │
│  │  │  │  │ Portfolio │  │ │  │ Portfolio │  │ │ │ Portfolio │ │  │   │  │
│  │  │  │  │   State   │  │ │  │   State   │  │ │ │   State   │ │  │   │  │
│  │  │  │  ├───────────┤  │ │  ├───────────┤  │ │ ├───────────┤ │  │   │  │
│  │  │  │  │  Metrics  │  │ │  │  Metrics  │  │ │ │  Metrics  │ │  │   │  │
│  │  │  │  └───────────┘  │ │  └───────────┘  │ │ └───────────┘ │  │   │  │
│  │  │  └─────────────────┘ └─────────────────┘ └──────────────┘  │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Granular Test Configurations                    │   │  │
│  │  │                                                              │   │  │
│  │  │  test_strategies:                                           │   │  │
│  │  │    - name: "baseline_adaptive"                              │   │  │
│  │  │      components:                                            │   │  │
│  │  │        - regime_classifier: "production_v1"                 │   │  │
│  │  │        - strategies: ["ma_optimal", "rsi_optimal"]          │   │  │
│  │  │        - ensemble_weights: "from_phase1"                    │   │  │
│  │  │                                                              │   │  │
│  │  │    - name: "enhanced_with_filters"                          │   │  │
│  │  │      components:                                            │   │  │
│  │  │        - regime_classifier: "production_v1"                 │   │  │
│  │  │        - strategies: ["ma_optimal", "rsi_optimal"]          │   │  │
│  │  │        - filters: ["counter_trend_veto", "volatility_adj"]  │   │  │
│  │  │        - ensemble_weights: "from_phase1"                    │   │  │
│  │  │                                                              │   │  │
│  │  │    - name: "analysis_discovered"                            │   │  │
│  │  │      components:                                            │   │  │
│  │  │        - custom_strategy: "time_based_parameter_switch"     │   │  │
│  │  │        - regime_aware: true                                 │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Test Results & Comparison                          │  │
│  │  • Side-by-side performance metrics                                  │  │
│  │  • Statistical significance testing                                  │  │
│  │  • Regime-specific performance breakdown                             │  │
│  │  • Final strategy selection for production                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Implementation Details

### YAML Configuration Example

```yaml
pipeline:
  name: "complete_strategy_development"
  
  phase_1_mining:
    data: "data/train_2020_2023.parquet"
    components:
      regime_classifier: "hmm_4_state"
      indicators: ["MA", "RSI", "ATR", "MACD"]
      parameter_grid: "configs/exhaustive_search.yaml"
    outputs:
      signals: true
      trades: true
      metrics: true
      
  phase_2_analysis:
    inputs: ["phase_1_mining.outputs"]
    notebooks: "analysis/exploration/"
    experiments:
      - counter_trend_filter
      - volatility_adjustment
      - time_based_switching
    constraints:
      data_access: "training_only"
      
  phase_3_test:
    data: "data/test_2024.parquet"
    strategies:
      - baseline_from_phase1
      - enhanced_with_filters
      - custom_discoveries
    execution:
      parallel: true
      containers: "isolated"
      shared_indicators: true
```

### Benefits of Complete Pipeline

1. **Scientific Rigor**: Clear separation of training, analysis, and testing
2. **Rich Exploration**: Analysis phase enables deep understanding
3. **Granular Control**: Every component configurable at every phase
4. **Fair Comparison**: Identical test conditions for all strategies
5. **No Data Snooping**: Strict isolation between training and test data

## Detailed Step 1: Grid Search with Regime Classification

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          STEP 1: GRID SEARCH                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐         ┌─────────────────────────────────────────┐    │
│  │  Market Data   │         │         Regime Classifier                │    │
│  │    Stream      │────────▶│  ┌─────────────────────────────────┐   │    │
│  └────────────────┘         │  │ Adds regime metadata to bars:   │   │    │
│                             │  │ - TRENDING_UP                    │   │    │
│                             │  │ - TRENDING_DOWN                  │   │    │
│                             │  │ - RANGE_BOUND                    │   │    │
│                             │  │ - HIGH_VOLATILITY                │   │    │
│                             │  └─────────────────────────────────┘   │    │
│                             └───────────────┬─────────────────────────┘    │
│                                             │                               │
│                                             ▼                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Shared Indicator Architecture                      │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Indicator Hub (Shared Computation)              │   │  │
│  │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │   │  │
│  │  │  │MA(5) │ │MA(10)│ │MA(15)│ │MA(20)│ │RSI(14)│ │RSI(21)│   │   │  │
│  │  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘   │   │  │
│  │  └─────┼────────┼───────┼───────┼───────┼────────┼──────────┘   │  │
│  │        └────────┴───────┴───────┴───────┴────────┴──────┐       │  │
│  │                                                          ▼       │  │
│  │  ┌─────────────────────────────────────────────────────────┐   │  │
│  │  │                 Event Bus (Broadcast)                    │   │  │
│  │  └─────┬───────┬───────┬───────┬───────┬───────┬──────────┘   │  │
│  │        ▼       ▼       ▼       ▼       ▼       ▼              │  │
│  │  ┌─────────────────────────────────────────────────────┐      │  │
│  │  │           Strategy Containers (100s-1000s)           │      │  │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │      │  │
│  │  │  │MA(5,10)│ │MA(5,15)│ │MA(10,20)│ │RSI+MA  │  ...  │      │  │
│  │  │  │+Regime │ │+Regime │ │+Regime │ │+Regime │       │      │  │
│  │  │  └────────┘ └────────┘ └────────┘ └────────┘       │      │  │
│  │  └──────────────────────────────────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         Output Storage                            │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  Results per Strategy (Streamed to Disk):                  │ │  │
│  │  │  - Strategy Parameters (MA periods, etc.)                  │ │  │
│  │  │  - Regime-Tagged Trades                                    │ │  │
│  │  │  - Performance Metrics (Sharpe, Returns, etc.)             │ │  │
│  │  │  - Per-Regime Performance Breakdown                        │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Detailed Step 2: Regime Analysis

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          STEP 2: REGIME ANALYSIS                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────┐                                                 │
│  │  Grid Search Results   │                                                 │
│  │  (All Strategy Trades) │                                                 │
│  └───────────┬────────────┘                                                 │
│              │                                                               │
│              ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        RegimeAnalyzer                                │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  For each regime type:                                      │   │  │
│  │  │  1. Filter trades by regime tag                             │   │  │
│  │  │  2. Group by strategy parameters                            │   │  │
│  │  │  3. Calculate regime-specific metrics                       │   │  │
│  │  │  4. Rank strategies within regime                           │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Regime-Optimal Parameters                       │   │  │
│  │  ├─────────────────┬──────────────────────────────────────────┤   │  │
│  │  │ TRENDING_UP     │ MA(5,20), RSI(14) - Sharpe: 2.1         │   │  │
│  │  │ TRENDING_DOWN   │ MA(10,20), RSI(21) - Sharpe: 1.8       │   │  │
│  │  │ RANGE_BOUND     │ MA(5,10), RSI(14) - Sharpe: 1.5        │   │  │
│  │  │ HIGH_VOLATILITY │ MA(15,20), RSI(21) - Sharpe: 1.2       │   │  │
│  │  └─────────────────┴──────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Regime Strategy Map                               │  │
│  │    (Defines which parameter set to use in each regime)              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Step 3: Signal-Based Weight Optimization

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   STEP 3: SIGNAL-BASED WEIGHT OPTIMIZATION                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐     ┌────────────────────────────────────┐         │
│  │ Regime Strategy    │     │   Captured Signals from Step 1      │         │
│  │      Map           │     │  (Regime-Tagged Signal Events)     │         │
│  └─────────┬──────────┘     └───────────┬────────────────────────┘         │
│            │                             │                                   │
│            └──────────┬──────────────────┘                                  │
│                       ▼                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Signal Storage Structure                          │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  Stored Signals by Regime (from SignalCollector):           │   │  │
│  │  │  - timestamp                                                 │   │  │
│  │  │  - symbol                                                    │   │  │
│  │  │  - signal_type                                               │   │  │
│  │  │  - price_at_signal                                           │   │  │
│  │  │  - ma_strength (component signal)                            │   │  │
│  │  │  - rsi_strength (component signal)                           │   │  │
│  │  │  - regime tag                                                │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              Signal-Based Weight Optimization Process                 │  │
│  │                                                                       │  │
│  │  For each regime:                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  1. Filter signals for current regime                        │   │  │
│  │  │  2. Initialize genetic optimizer                             │   │  │
│  │  │  3. For each weight combination candidate:                  │   │  │
│  │  │     a. Apply weights to component signals                   │   │  │
│  │  │     b. Generate new weighted signals                        │   │  │
│  │  │     c. Replay through existing Portfolio (reset)            │   │  │
│  │  │     d. Calculate fitness (Sharpe, returns, etc.)            │   │  │
│  │  │  4. Evolve to find optimal weights                          │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │                RegimeSignalReplayer                          │   │  │
│  │  │                                                              │   │  │
│  │  │  replay_regime_signals(regime, signals, weights):           │   │  │
│  │  │    1. Reset Portfolio to initial state                      │   │  │
│  │  │    2. For each stored signal in temporal order:             │   │  │
│  │  │       - Apply new weights to component strengths            │   │  │
│  │  │       - Generate weighted signal event                      │   │  │
│  │  │       - Publish to EventBus → Portfolio                     │   │  │
│  │  │    3. Return portfolio performance metrics                  │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │         Weight Application Example (per signal):             │   │  │
│  │  │                                                              │   │  │
│  │  │  Original: ma_strength=0.8, rsi_strength=0.4                │   │  │
│  │  │  Weights:  ma_weight=0.7, rsi_weight=0.3                    │   │  │
│  │  │  Combined: (0.8 * 0.7) + (0.4 * 0.3) = 0.68                │   │  │
│  │  │  Signal:   BUY if > 0.6, SELL if < -0.6, else HOLD         │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Optimized Ensemble Weights by Regime            │   │  │
│  │  ├─────────────────┬──────────────────────────────────────────┤   │  │
│  │  │ TRENDING_UP     │ MA: 0.65, RSI: 0.35 (Sharpe: 2.3)       │   │  │
│  │  │ TRENDING_DOWN   │ MA: 0.30, RSI: 0.70 (Sharpe: 1.9)       │   │  │
│  │  │ RANGE_BOUND     │ MA: 0.45, RSI: 0.55 (Sharpe: 1.7)       │   │  │
│  │  │ HIGH_VOLATILITY │ MA: 0.20, RSI: 0.80 (Sharpe: 1.4)       │   │  │
│  │  └─────────────────┴──────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              Key Efficiency Gains                                     │  │
│  │  • No indicator recalculation                                        │  │
│  │  • No market data replay                                             │  │
│  │  • 25-70% smaller datasets (regime-filtered)                         │  │
│  │  • 100-1000x faster than full backtests                              │  │
│  │  • Parallel regime optimization possible                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Step 4: Test Execution

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         STEP 4: TEST EXECUTION                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐     ┌────────────────────────┐                     │
│  │  Test Dataset      │     │  Adaptive Strategy     │                     │
│  │  (Unseen Data)     │     │    Configuration       │                     │
│  └─────────┬──────────┘     └───────────┬────────────┘                     │
│            │                             │                                   │
│            └──────────┬──────────────────┘                                  │
│                       ▼                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Adaptive Strategy Execution                        │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐                                                │  │
│  │  │ Market Data     │                                                │  │
│  │  └────────┬────────┘                                                │  │
│  │           ▼                                                          │  │
│  │  ┌─────────────────┐     ┌──────────────────────────────┐         │  │
│  │  │Regime Classifier│────▶│  Current Regime: TRENDING_UP  │         │  │
│  │  └─────────────────┘     └──────────────┬───────────────┘         │  │
│  │                                          │                          │  │
│  │                                          ▼                          │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │              Strategy Selector & Executor                    │ │  │
│  │  │                                                              │ │  │
│  │  │  1. Identify current regime                                  │ │  │
│  │  │  2. Activate regime-optimal strategies                       │ │  │
│  │  │  3. Apply ensemble weights for current regime                │ │  │
│  │  │  4. Generate combined signal (100% investment assumption)    │ │  │
│  │  │  5. Track performance metrics                                │ │  │
│  │  └──────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │                  Test Results Storage                         │ │  │
│  │  │  - Overall performance metrics                                │ │  │
│  │  │  - Per-regime performance breakdown                           │ │  │
│  │  │  - Regime transition analysis                                 │ │  │
│  │  │  - Strategy activation frequency                              │ │  │
│  │  └──────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Data Flow Characteristics

### 1. **Regime-Aware Processing**
- Every bar is tagged with regime classification
- All performance tracking is regime-specific
- Strategies are optimized for specific market conditions

### 2. **Shared Computation**
- Indicators calculated once, used by many strategies
- Significant reduction in computational overhead
- Enables testing of thousands of parameter combinations

### 3. **100% Investment Assumption**
- Signals are evaluated as if fully invested
- Simplifies optimization to focus on signal quality
- Position sizing can be added later

### 4. **Progressive Refinement**
- Grid search finds good parameters
- Regime analysis finds best parameters per condition
- Ensemble optimization finds optimal combinations
- Test validates the adaptive approach

### 5. **Memory Efficiency**
- Results streamed to disk during grid search
- Only top performers kept in memory
- Batch processing if needed for very large parameter spaces

## Implementation Notes

1. **Regime Classifier Integration**
   - Runs inline with market data processing
   - Adds metadata to every bar before indicator calculation
   - Must be deterministic for reproducibility

2. **Storage Strategy**
   - Grid search results: Parquet files partitioned by batch
   - Regime analysis: Summary statistics in memory
   - Final configuration: JSON/YAML for easy loading

3. **Performance Tracking**
   - Every trade tagged with active regime
   - Metrics calculated both overall and per-regime
   - Regime transition points tracked for analysis

4. **Ensemble Execution**
   - Strategies can run in parallel (same shared indicators)
   - Weights applied to signals before combining
   - Single combined position per timestamp
