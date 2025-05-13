# Risk Limit Composition

This document outlines the design for the Risk Limit Composition system in ADMF-Trader. It addresses item 5.2 in the IMPROVEMENTS.md document, focusing on creating formal risk constraint composition patterns, clear precedence rules for conflicting risk limits, and validation mechanisms for risk limit consistency.

## 1. Overview

The Risk Limit Composition system enables the application and enforcement of multiple risk constraints in a coordinated and consistent manner. It provides a formal framework for combining risk limits, resolving conflicts between them, and validating their consistency.

## 2. Core Architecture

The Risk Limit Composition framework consists of these key components:

```
RiskLimitComposition
  ├── RiskLimitRegistry
  ├── CompositionStrategies
  ├── PrecedenceRules
  ├── RiskConstraintValidator
  └── RiskLimitEvaluator
```

### 2.1 Risk Limit Registry

The Risk Limit Registry manages risk limits and their relationships:

```python
class RiskLimitRegistry:
    """
    Central registry for all risk limits in the system.
    
    Maintains relationships between risk limits and provides
    access to limit configurations.
    """
    
    def __init__(self):
        self.limits = {}  # type_id -> RiskLimit
        self.limit_metadata = {}  # type_id -> metadata
        self.limit_relationships = {}  # (limit_id1, limit_id2) -> relationship
        
    def register_limit(self, limit, metadata=None):
        """Register a risk limit with optional metadata."""
        limit_id = limit.get_id()
        self.limits[limit_id] = limit
        self.limit_metadata[limit_id] = metadata or {}
        
    def set_relationship(self, limit_id1, limit_id2, relationship):
        """Set relationship between two limits."""
        self.limit_relationships[(limit_id1, limit_id2)] = relationship
        
    def get_relationship(self, limit_id1, limit_id2):
        """Get relationship between two limits."""
        return self.limit_relationships.get((limit_id1, limit_id2))
        
    def get_limits_by_type(self, limit_type):
        """Get all limits of a specific type."""
        return [l for l in self.limits.values() if l.get_type() == limit_type]
        
    def get_limit_metadata(self, limit_id):
        """Get metadata for a limit."""
        return self.limit_metadata.get(limit_id, {})
```

### 2.2 Enhanced Risk Limit Interface

All risk limits will implement this enhanced interface that supports composition:

```python
class RiskLimitBase:
    """
    Base class for all risk limits.
    
    Provides interface for checking limits and supporting composition.
    """
    
    def __init__(self, parameters=None):
        """Initialize with parameters."""
        self.parameters = parameters or {}
        self.limit_id = str(uuid.uuid4())
        self.limit_type = self.__class__.__name__
        self.priority = self.parameters.get('priority', 50)  # Default priority
        
    def get_id(self):
        """Get unique identifier for this limit."""
        return self.limit_id
        
    def get_type(self):
        """Get limit type."""
        return self.limit_type
        
    def get_priority(self):
        """Get limit priority."""
        return self.priority
        
    def check(self, signal, quantity, portfolio):
        """
        Check if signal passes the risk limit.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            portfolio: Portfolio object
            
        Returns:
            bool: Whether signal passes the risk limit
        """
        raise NotImplementedError
        
    def explain_violation(self, signal, quantity, portfolio):
        """
        Explain why a limit was violated.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            portfolio: Portfolio object
            
        Returns:
            str: Explanation of violation
        """
        return f"Risk limit {self.limit_type} was violated"
        
    def modify_quantity(self, signal, quantity, portfolio):
        """
        Modify quantity to conform to risk limit.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            portfolio: Portfolio object
            
        Returns:
            int: Modified quantity that would pass this limit
        """
        return 0  # Default implementation rejects trade completely
```

## 3. Composition Strategies

Composition strategies define how multiple risk limits are combined and evaluated.

### 3.1 Composition Strategy Interface

```python
class CompositionStrategy:
    """
    Strategy for composing multiple risk limits.
    
    Defines how multiple risk limits are combined and evaluated.
    """
    
    def compose(self, risk_limits, signal, quantity, portfolio):
        """
        Compose multiple risk limits.
        
        Args:
            risk_limits: List of risk limits to compose
            signal: Signal data dictionary
            quantity: Calculated position size
            portfolio: Portfolio object
            
        Returns:
            CompositionResult: Result of composition
        """
        raise NotImplementedError
```

### 3.2 Composition Result

Standardizes the result of limit composition:

```python
class CompositionResult:
    """Result of composing multiple risk limits."""
    
    def __init__(self):
        self.passed = False
        self.modified_quantity = None
        self.violations = []
        self.limit_results = {}  # limit_id -> individual result
        
    def add_limit_result(self, limit_id, passed, explanation=None):
        """Add individual limit result."""
        self.limit_results[limit_id] = {
            'passed': passed,
            'explanation': explanation
        }
        
        if not passed and explanation:
            self.violations.append(explanation)
            
    def set_modified_quantity(self, quantity):
        """Set modified quantity that would pass limits."""
        self.modified_quantity = quantity
        
    def set_passed(self, passed):
        """Set overall pass/fail result."""
        self.passed = passed
        
    def get_result_summary(self):
        """Get summary of composition result."""
        return {
            'passed': self.passed,
            'modified_quantity': self.modified_quantity,
            'violations': self.violations,
            'limit_results': self.limit_results
        }
```

### 3.3 Concrete Composition Strategies

#### 3.3.1 All-Pass Strategy

Requires all limits to pass for the signal to be accepted:

```python
class AllPassStrategy(CompositionStrategy):
    """Strategy requiring all limits to pass."""
    
    def compose(self, risk_limits, signal, quantity, portfolio):
        """Require all limits to pass."""
        result = CompositionResult()
        all_passed = True
        
        for limit in risk_limits:
            limit_id = limit.get_id()
            passed = limit.check(signal, quantity, portfolio)
            
            explanation = None
            if not passed:
                all_passed = False
                explanation = limit.explain_violation(signal, quantity, portfolio)
                
            result.add_limit_result(limit_id, passed, explanation)
            
        result.set_passed(all_passed)
        
        # If not all passed, find most restrictive modification
        if not all_passed:
            modified_quantities = []
            
            for limit in risk_limits:
                if not limit.check(signal, quantity, portfolio):
                    mod_qty = limit.modify_quantity(signal, quantity, portfolio)
                    modified_quantities.append(mod_qty)
            
            if modified_quantities:
                # Most restrictive is the smallest absolute quantity
                most_restrictive = min(modified_quantities, key=abs)
                result.set_modified_quantity(most_restrictive)
                
        return result
```

#### 3.3.2 Weighted Strategy

Uses weighted voting among limits to determine the outcome:

```python
class WeightedStrategy(CompositionStrategy):
    """Strategy using weighted voting among limits."""
    
    def compose(self, risk_limits, signal, quantity, portfolio):
        """Use weighted voting to decide."""
        result = CompositionResult()
        total_weight = sum(limit.get_priority() for limit in risk_limits)
        weighted_votes = 0
        
        for limit in risk_limits:
            limit_id = limit.get_id()
            passed = limit.check(signal, quantity, portfolio)
            weight = limit.get_priority() / total_weight if total_weight > 0 else 0
            
            explanation = None
            if not passed:
                explanation = limit.explain_violation(signal, quantity, portfolio)
            else:
                weighted_votes += weight
                
            result.add_limit_result(limit_id, passed, explanation)
            
        # Determine overall result
        threshold = 0.5  # Configurable
        passed = weighted_votes >= threshold
        result.set_passed(passed)
        
        # Find best modification if needed
        if not passed:
            modified_quantities = []
            
            for limit in risk_limits:
                if not limit.check(signal, quantity, portfolio):
                    mod_qty = limit.modify_quantity(signal, quantity, portfolio)
                    modified_quantities.append(mod_qty)
            
            if modified_quantities:
                # Use weighted average of modifications
                weights = [limit.get_priority() for limit in risk_limits 
                          if not limit.check(signal, quantity, portfolio)]
                total_mod_weight = sum(weights)
                
                if total_mod_weight > 0:
                    weighted_mod = sum(q * (w/total_mod_weight) for q, w in zip(modified_quantities, weights))
                    result.set_modified_quantity(int(weighted_mod))
                    
        return result
```

#### 3.3.3 Priority-Based Strategy

Evaluates limits in priority order, with highest priority limits taking precedence:

```python
class PriorityBasedStrategy(CompositionStrategy):
    """Strategy using priority ordering of limits."""
    
    def compose(self, risk_limits, signal, quantity, portfolio):
        """Evaluate limits in priority order."""
        result = CompositionResult()
        
        # Sort limits by priority (highest first)
        ordered_limits = sorted(risk_limits, key=lambda x: x.get_priority(), reverse=True)
        highest_priority_failure = None
        
        for limit in ordered_limits:
            limit_id = limit.get_id()
            passed = limit.check(signal, quantity, portfolio)
            
            explanation = None
            if not passed:
                explanation = limit.explain_violation(signal, quantity, portfolio)
                
                if highest_priority_failure is None:
                    highest_priority_failure = limit
                    
            result.add_limit_result(limit_id, passed, explanation)
            
        # Overall result based on highest priority limit
        passed = highest_priority_failure is None
        result.set_passed(passed)
        
        # Use modification from highest priority failing limit
        if not passed:
            mod_qty = highest_priority_failure.modify_quantity(signal, quantity, portfolio)
            result.set_modified_quantity(mod_qty)
            
        return result
```

## 4. Precedence Rules

Precedence rules define how conflicts between limits are resolved.

### 4.1 Precedence Rule Interface

```python
class PrecedenceRule:
    """
    Rule for resolving conflicts between risk limits.
    
    Determines which limit takes precedence when multiple limits conflict.
    """
    
    def check_precedence(self, limit1, limit2):
        """
        Check which limit takes precedence.
        
        Args:
            limit1: First RiskLimit
            limit2: Second RiskLimit
            
        Returns:
            int: Positive if limit1 takes precedence, negative if limit2,
                 zero if equal precedence
        """
        raise NotImplementedError
```

### 4.2 Concrete Precedence Rules

#### 4.2.1 Priority-Based Precedence

Precedence based on limit priority:

```python
class PriorityBasedPrecedence(PrecedenceRule):
    """Precedence based on limit priority."""
    
    def check_precedence(self, limit1, limit2):
        """Compare limit priorities."""
        return limit1.get_priority() - limit2.get_priority()
```

#### 4.2.2 Type-Based Precedence

Precedence based on limit type hierarchy:

```python
class TypeBasedPrecedence(PrecedenceRule):
    """Precedence based on limit type hierarchy."""
    
    def __init__(self, type_hierarchy):
        """
        Initialize with type hierarchy.
        
        Args:
            type_hierarchy: Dict mapping limit types to precedence values
                            (higher values = higher precedence)
        """
        self.type_hierarchy = type_hierarchy
        
    def check_precedence(self, limit1, limit2):
        """Compare types in hierarchy."""
        type1 = limit1.get_type()
        type2 = limit2.get_type()
        
        rank1 = self.type_hierarchy.get(type1, 0)
        rank2 = self.type_hierarchy.get(type2, 0)
        
        return rank1 - rank2
```

#### 4.2.3 Explicit Precedence Rule

Precedence defined by explicit relationships:

```python
class ExplicitPrecedenceRule(PrecedenceRule):
    """Precedence defined by explicit relationships."""
    
    def __init__(self, registry):
        """
        Initialize with registry.
        
        Args:
            registry: RiskLimitRegistry containing relationships
        """
        self.registry = registry
        
    def check_precedence(self, limit1, limit2):
        """Check explicit relationship."""
        limit_id1 = limit1.get_id()
        limit_id2 = limit2.get_id()
        
        relationship = self.registry.get_relationship(limit_id1, limit_id2)
        
        if relationship == 'dominates':
            return 1
        elif relationship == 'submits_to':
            return -1
        else:
            # Default to priority comparison
            return limit1.get_priority() - limit2.get_priority()
```

## 5. Risk Constraint Validator

The validator ensures consistency and validity of risk limits.

### 5.1 Validator Interface

```python
class RiskConstraintValidator:
    """
    Validator for risk constraints.
    
    Checks consistency and validity of risk limits.
    """
    
    def validate_limits(self, limits, portfolio_context=None):
        """
        Validate a set of risk limits.
        
        Args:
            limits: List of risk limits to validate
            portfolio_context: Optional portfolio context for validation
            
        Returns:
            ValidationResult: Result of validation
        """
        raise NotImplementedError
```

### 5.2 Validation Result

Standardizes validation results:

```python
class ValidationResult:
    """Result of validating risk limits."""
    
    def __init__(self):
        self.valid = True
        self.issues = []
        self.warnings = []
        
    def add_issue(self, issue, is_warning=False):
        """Add validation issue."""
        if is_warning:
            self.warnings.append(issue)
        else:
            self.valid = False
            self.issues.append(issue)
            
    def get_result_summary(self):
        """Get summary of validation result."""
        return {
            'valid': self.valid,
            'issues': self.issues,
            'warnings': self.warnings
        }
```

### 5.3 Concrete Validators

#### 5.3.1 Basic Constraint Validator

Performs basic validation of risk constraints:

```python
class BasicConstraintValidator(RiskConstraintValidator):
    """Basic validation for risk constraints."""
    
    def validate_limits(self, limits, portfolio_context=None):
        """Validate limits for basic consistency."""
        result = ValidationResult()
        
        # Check for duplicate limits
        limit_types = {}
        for limit in limits:
            limit_type = limit.get_type()
            
            if limit_type in limit_types:
                result.add_issue(
                    f"Duplicate risk limit type: {limit_type}",
                    is_warning=True
                )
                
            limit_types[limit_type] = True
            
        # Check for parameter consistency
        for limit in limits:
            params = limit.parameters
            
            # Check for required parameters
            if 'max_position' in params and params['max_position'] <= 0:
                result.add_issue(
                    f"Invalid max_position parameter in {limit.get_type()}: {params['max_position']}"
                )
                
            if 'max_exposure' in params and params['max_exposure'] <= 0:
                result.add_issue(
                    f"Invalid max_exposure parameter in {limit.get_type()}: {params['max_exposure']}"
                )
                
        return result
```

#### 5.3.2 Conflict Detection Validator

Detects conflicts between risk constraints:

```python
class ConflictDetectionValidator(RiskConstraintValidator):
    """Validates for conflicts between risk constraints."""
    
    def validate_limits(self, limits, portfolio_context=None):
        """Detect conflicts between limits."""
        result = ValidationResult()
        
        # Group limits by type
        limits_by_type = {}
        for limit in limits:
            limit_type = limit.get_type()
            
            if limit_type not in limits_by_type:
                limits_by_type[limit_type] = []
                
            limits_by_type[limit_type].append(limit)
            
        # Check position size limits for conflicts
        position_limits = limits_by_type.get('MaxPositionSizeLimit', [])
        if len(position_limits) > 1:
            # Check for conflicting maximum position sizes
            max_sizes = [l.parameters.get('max_position') for l in position_limits]
            
            if max(max_sizes) > 2 * min(max_sizes):
                result.add_issue(
                    f"Potentially conflicting position size limits: {max_sizes}",
                    is_warning=True
                )
                
        # Check exposure vs position limits
        exposure_limits = limits_by_type.get('MaxExposureLimit', [])
        
        if portfolio_context and exposure_limits and position_limits:
            # Estimate if exposure limits are consistent with position limits
            portfolio_value = portfolio_context.get_portfolio_value()
            
            for exp_limit in exposure_limits:
                max_exposure_pct = exp_limit.parameters.get('max_exposure', 0)
                max_exposure_val = portfolio_value * (max_exposure_pct / 100.0)
                
                for pos_limit in position_limits:
                    max_position = pos_limit.parameters.get('max_position', 0)
                    
                    # Assume an average price of 100 for simplicity
                    avg_price = 100
                    max_position_value = max_position * avg_price
                    
                    if max_position_value > max_exposure_val * 1.5:
                        result.add_issue(
                            f"Position limit ({max_position} shares) may exceed "
                            f"exposure limit ({max_exposure_pct}%)",
                            is_warning=True
                        )
                        
        return result
```

#### 5.3.3 Composite Validator

Combines multiple validators:

```python
class CompositeValidator(RiskConstraintValidator):
    """Combines multiple validators."""
    
    def __init__(self, validators):
        """
        Initialize with validators.
        
        Args:
            validators: List of RiskConstraintValidator instances
        """
        self.validators = validators
        
    def validate_limits(self, limits, portfolio_context=None):
        """Run all validators."""
        result = ValidationResult()
        
        for validator in self.validators:
            sub_result = validator.validate_limits(limits, portfolio_context)
            
            # Combine results
            for issue in sub_result.issues:
                result.add_issue(issue)
                
            for warning in sub_result.warnings:
                result.add_issue(warning, is_warning=True)
                
        return result
```

## 6. Risk Limit Evaluator

The Risk Limit Evaluator ties together the composition framework:

```python
class RiskLimitEvaluator:
    """
    Evaluates risk limits using composition strategies.
    
    Central component for applying risk constraints to trades.
    """
    
    def __init__(self, registry, composition_strategy, validator=None):
        """
        Initialize evaluator.
        
        Args:
            registry: RiskLimitRegistry
            composition_strategy: CompositionStrategy
            validator: Optional RiskConstraintValidator
        """
        self.registry = registry
        self.composition_strategy = composition_strategy
        self.validator = validator
        
    def validate_limits(self, portfolio_context=None):
        """
        Validate all registered limits.
        
        Args:
            portfolio_context: Optional portfolio for validation context
            
        Returns:
            ValidationResult if validator exists, None otherwise
        """
        if self.validator:
            limits = list(self.registry.limits.values())
            return self.validator.validate_limits(limits, portfolio_context)
        return None
        
    def evaluate_signal(self, signal, quantity, portfolio):
        """
        Evaluate a signal against all applicable risk limits.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            portfolio: Portfolio object
            
        Returns:
            CompositionResult: Result of evaluation
        """
        # Get applicable limits
        symbol = signal.get('symbol')
        limits = []
        
        # Filter applicable limits
        for limit_id, limit in self.registry.limits.items():
            metadata = self.registry.get_limit_metadata(limit_id)
            
            # Get applicable symbols for this limit
            applicable_symbols = metadata.get('symbols', None)
            
            # If symbols specified, check if this symbol applies
            if applicable_symbols is not None and symbol not in applicable_symbols:
                continue
                
            limits.append(limit)
            
        # Compose limits
        result = self.composition_strategy.compose(limits, signal, quantity, portfolio)
        
        return result
```

## 7. Integration with Risk Manager

The Risk Limit Composition framework integrates into the Risk Manager component:

```python
class RiskManager(Component):
    """
    Risk manager with enhanced limit composition.
    
    Converts signals to orders with position sizing and risk limits.
    """
    
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Get portfolio manager
        self.portfolio = self._get_dependency(context, 'portfolio', required=True)
        
        # Set up event subscriptions
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.SIGNAL, self.on_signal)
        
        # Create position sizers
        self._create_position_sizers()
        
        # Create risk limit registry
        self.risk_limit_registry = RiskLimitRegistry()
        
        # Create risk limits
        self._create_risk_limits()
        
        # Define composition strategy
        composition_strategy_type = self.parameters.get('composition_strategy', 'all_pass')
        
        if composition_strategy_type == 'all_pass':
            self.composition_strategy = AllPassStrategy()
        elif composition_strategy_type == 'weighted':
            self.composition_strategy = WeightedStrategy()
        elif composition_strategy_type == 'priority':
            self.composition_strategy = PriorityBasedStrategy()
            
        # Create validator
        basic_validator = BasicConstraintValidator()
        conflict_validator = ConflictDetectionValidator()
        self.validator = CompositeValidator([basic_validator, conflict_validator])
        
        # Create evaluator
        self.risk_evaluator = RiskLimitEvaluator(
            self.risk_limit_registry,
            self.composition_strategy,
            self.validator
        )
        
        # Validate limits
        validation_result = self.risk_evaluator.validate_limits(self.portfolio)
        
        if validation_result and not validation_result.valid:
            self.logger.warning(f"Risk limit validation failed: {validation_result.get_result_summary()}")
    
    def _check_risk_limits(self, signal, quantity):
        """
        Check if signal passes all risk limits using composition.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            
        Returns:
            bool: Whether signal passes all limits
        """
        # Evaluate signal against composition rules
        result = self.risk_evaluator.evaluate_signal(signal, quantity, self.portfolio)
        
        # If signal failed but suggested modification exists
        if not result.passed and result.modified_quantity is not None:
            # Update quantity in signal (to be used by order creation)
            signal['_original_quantity'] = quantity
            signal['_modified_quantity'] = result.modified_quantity
            
            # Log modifications
            self.logger.info(
                f"Signal quantity modified from {quantity} to {result.modified_quantity} "
                f"due to risk limits. Violations: {result.violations}"
            )
            
            return True  # Allow to proceed with modified quantity
            
        return result.passed
```

## 8. Example Configuration

Example risk limit composition configuration:

```json
{
  "risk_limits": [
    {
      "type": "position",
      "max_position": 1000,
      "priority": 80,
      "symbols": ["AAPL", "MSFT", "GOOGL"]
    },
    {
      "type": "position",
      "max_position": 500,
      "priority": 80,
      "symbols": ["AMZN", "FB", "NFLX"]
    },
    {
      "type": "exposure",
      "max_exposure": 20.0,
      "priority": 90
    },
    {
      "type": "drawdown",
      "max_drawdown": 15.0,
      "reduce_threshold": 10.0,
      "reduction_factor": 0.5,
      "priority": 95
    }
  ],
  "precedence_rules": [
    {
      "dominant": "DrawdownLimit",
      "submissive": "PositionLimit"
    },
    {
      "dominant": "DrawdownLimit",
      "submissive": "ExposureLimit"
    }
  ],
  "composition_strategy": "priority"
}
```

## 9. Best Practices

### 9.1 Limit Configuration

- Assign clear priorities to all risk limits
- Group related limits logically (e.g., by asset class or strategy)
- Set explicit precedence rules for critical limits
- Document the intent behind each limit
- Validate limit consistency when changing parameters

### 9.2 Composition Strategy Selection

- **AllPassStrategy**: Use for strict risk enforcement where all limits must be satisfied
- **WeightedStrategy**: Use for flexible risk management with limit importance weighting
- **PriorityBasedStrategy**: Use when clear hierarchy exists between limit types

### 9.3 Risk Limit Validation

- Run validation regularly, not just at startup
- Log validation warnings, even if not critical
- Verify limits against expected market scenarios
- Test limit composition with historical data

## 10. Implementation Approach

The implementation will follow these steps:

1. Enhance the existing risk limit classes with composition support
2. Implement the core composition infrastructure (registry, strategies, precedence rules)
3. Create the validation framework
4. Integrate with the risk manager
5. Add configuration and documentation
6. Develop comprehensive tests

## 11. Testing Strategy

The testing strategy will include:

1. **Unit Tests**:
   - Test each risk limit type individually
   - Verify each composition strategy
   - Test precedence rules
   - Validate constraint validation

2. **Integration Tests**:
   - Test end-to-end signal to order flow with risk limits
   - Verify composition behavior with multiple limits
   - Test conflict resolution with different strategies

3. **Edge Case Tests**:
   - Test extreme market conditions
   - Verify behavior with conflicting limits
   - Test performance with many simultaneous limits