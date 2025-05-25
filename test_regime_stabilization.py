#!/usr/bin/env python3
"""
Test to understand regime stabilization behavior
"""

def simulate_regime_stabilization():
    """Simulate the regime detector's stabilization logic"""
    
    # Scenario 1: Cold start (production-like)
    print("=== SCENARIO 1: Cold Start (Production) ===")
    current_classification = None
    pending_regime = None
    pending_duration = 0
    min_regime_duration = 2
    current_regime_duration = 0
    
    # Simulate bars with regime detection
    detections = ['default', 'default', 'default', 'trending_up_volatile', 'trending_up_volatile']
    
    for i, detected in enumerate(detections):
        print(f"\nBar {i}: Detected regime = {detected}")
        
        # Apply stabilization (simplified from RegimeDetector._apply_stabilization)
        if current_classification is None:
            # First classification - no stabilization needed
            final_regime = detected
            current_classification = detected
            current_regime_duration = 1
            print(f"  → First classification, setting to: {final_regime}")
        elif detected == current_classification:
            # Same regime continues
            current_regime_duration += 1
            final_regime = current_classification
            print(f"  → Same regime continues (duration: {current_regime_duration})")
        else:
            # Different regime detected
            if pending_regime == detected:
                # Continue pending
                pending_duration += 1
                print(f"  → Pending regime {pending_regime} continues (duration: {pending_duration})")
                
                if pending_duration >= min_regime_duration:
                    # Switch to pending regime
                    final_regime = pending_regime
                    current_classification = pending_regime
                    current_regime_duration = pending_duration
                    pending_regime = None
                    pending_duration = 0
                    print(f"  → REGIME SWITCH to {final_regime} after meeting min_duration")
                else:
                    final_regime = current_classification
                    print(f"  → Staying in {current_classification}, pending needs more duration")
            else:
                # New pending regime
                pending_regime = detected
                pending_duration = 1
                final_regime = current_classification
                print(f"  → New pending regime: {pending_regime}, staying in {current_classification}")
        
        print(f"  Final regime: {final_regime}")
    
    print("\n" + "="*50)
    
    # Scenario 2: Warm start (optimizer-like, already in a regime)
    print("\n=== SCENARIO 2: Warm Start (Optimizer) ===")
    current_classification = 'ranging_low_vol'  # Pre-existing regime
    pending_regime = None
    pending_duration = 0
    current_regime_duration = 10  # Already stable
    
    print(f"Starting with pre-existing regime: {current_classification} (duration: {current_regime_duration})")
    
    for i, detected in enumerate(detections):
        print(f"\nBar {i}: Detected regime = {detected}")
        
        # Apply stabilization
        if detected == current_classification:
            current_regime_duration += 1
            final_regime = current_classification
            print(f"  → Same regime continues (duration: {current_regime_duration})")
        else:
            if pending_regime == detected:
                pending_duration += 1
                print(f"  → Pending regime {pending_regime} continues (duration: {pending_duration})")
                
                if pending_duration >= min_regime_duration:
                    final_regime = pending_regime
                    current_classification = pending_regime
                    current_regime_duration = pending_duration
                    pending_regime = None
                    pending_duration = 0
                    print(f"  → REGIME SWITCH to {final_regime} after meeting min_duration")
                else:
                    final_regime = current_classification
                    print(f"  → Staying in {current_classification}, pending needs more duration")
            else:
                pending_regime = detected
                pending_duration = 1
                final_regime = current_classification
                print(f"  → New pending regime: {pending_regime}, staying in {current_classification}")
        
        print(f"  Final regime: {final_regime}")

if __name__ == "__main__":
    simulate_regime_stabilization()