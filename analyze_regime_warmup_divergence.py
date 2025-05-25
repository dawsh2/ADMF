#!/usr/bin/env python3
"""
Analyze the regime detector warmup period divergence between OOS and production runs.
"""

def analyze_warmup_impact():
    """
    Analyze how the 200-bar warmup period affects regime detection.
    """
    print("="*60)
    print("REGIME DETECTOR WARMUP ANALYSIS")
    print("="*60)
    
    # With train_ratio=0.8 on 1000 bars
    total_bars = 1000
    train_ratio = 0.8
    train_size = int(train_ratio * total_bars)  # 800 bars
    test_size = total_bars - train_size  # 200 bars
    
    print(f"\nDataset Analysis:")
    print(f"- Total bars: {total_bars}")
    print(f"- Train bars: {train_size} (bars 0-799)")
    print(f"- Test bars: {test_size} (bars 800-999)")
    
    print(f"\nRegime Detector Configuration:")
    print(f"- MA Trend: 50/200 periods")
    print(f"- ATR: 20 periods")
    print(f"- RSI: 14 periods")
    print(f"- Min regime duration: 3 bars")
    
    print(f"\nWarmup Impact on Test Data:")
    print(f"- MA Trend needs 200 bars to become ready")
    print(f"- Test data has only {test_size} bars")
    print(f"- Therefore: MA Trend NEVER becomes ready during test period!")
    
    print(f"\nThis explains the divergence:")
    print(f"1. In optimization's OOS test:")
    print(f"   - RegimeDetector runs on full dataset (1000 bars)")
    print(f"   - MA Trend becomes ready at bar 200")
    print(f"   - Can detect all 4 regimes during bars 200-999")
    print(f"   - OOS test evaluates performance on bars 800-999")
    print(f"   - But regime detection has been active since bar 200")
    
    print(f"\n2. In production run on test data:")
    print(f"   - RegimeDetector starts fresh at bar 800")
    print(f"   - Only sees 200 bars total (800-999)")
    print(f"   - MA Trend never becomes ready")
    print(f"   - Stays in 'default' regime entire time")
    print(f"   - Different regime = different trading behavior = different results")
    
    print(f"\nRoot Cause:")
    print(f"The optimizer's OOS test has 'memory' from the training period,")
    print(f"while the production run starts with a cold RegimeDetector.")
    
    print(f"\nSOLUTION OPTIONS:")
    print(f"1. Cold Start in Optimizer: Reset RegimeDetector before OOS test")
    print(f"2. Warm Start in Production: Pre-feed training data to warm up indicators")
    print(f"3. Shorter Indicators: Use periods that fit within test data (e.g., 20/50)")
    print(f"4. Extended Test Data: Include warmup bars in test dataset")
    
    print(f"\nRECOMMENDATION:")
    print(f"Implement Option 1 - Cold Start in Optimizer's OOS test")
    print(f"This ensures both runs start from the same state.")

if __name__ == "__main__":
    analyze_warmup_impact()