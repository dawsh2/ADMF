#!/usr/bin/env python3
"""Generate all valid weight combinations for 4 rules that sum to 1.0"""

def generate_weight_combinations():
    """Generate all combinations of weights [0, 0.2, 0.4, 0.6, 0.8, 1.0] that sum to 1.0"""
    weights = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    valid_combinations = []
    
    # Generate all possible combinations
    for w1 in weights:
        for w2 in weights:
            for w3 in weights:
                for w4 in weights:
                    # Check if weights sum to 1.0 (with floating point tolerance)
                    if abs(w1 + w2 + w3 + w4 - 1.0) < 0.001:
                        valid_combinations.append([w1, w2, w3, w4])
    
    # Remove duplicates and sort
    unique_combinations = []
    for combo in valid_combinations:
        if combo not in unique_combinations:
            unique_combinations.append(combo)
    
    return unique_combinations

if __name__ == "__main__":
    combinations = generate_weight_combinations()
    
    print(f"Total valid weight combinations: {len(combinations)}")
    print("\nAll combinations:")
    for i, combo in enumerate(combinations):
        print(f"        - [{combo[0]}, {combo[1]}, {combo[2]}, {combo[3]}]")
        
    # Show some interesting subsets
    print("\n\nBalanced combinations (no zeros):")
    balanced = [c for c in combinations if 0 not in c]
    for combo in balanced[:10]:
        print(f"        - [{combo[0]}, {combo[1]}, {combo[2]}, {combo[3]}]")
    
    print("\n\nSingle rule dominant (one weight >= 0.6):")
    dominant = [c for c in combinations if max(c) >= 0.6]
    for combo in dominant[:10]:
        print(f"        - [{combo[0]}, {combo[1]}, {combo[2]}, {combo[3]}]")