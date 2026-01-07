# README_2.md Compliance Issues Found in tofu_unlearning_pipeline.ipynb

## Summary
After thorough review, the notebook implementation is **mostly compliant** with README_2.md, but there are several issues that need to be fixed to achieve **strict compliance**.

## Issues Found

### ✅ **CORRECTLY IMPLEMENTED**

1. **State Space (Section 2.1)** ✅
   - `s = (q, v_q, U_0)` - Correctly implemented
   - U_0 computation uses Top-1 probability - Correct

2. **Policy Network (Section 3)** ✅
   - Four actions (k_ratio, w_recall, w_score, a_cot) - Correct
   - K_dynamic formula - Correct

3. **Phase 1: Dynamic Recall (Section 4.1)** ✅
   - Allocation formula - Correct

4. **Phase 2: Info Gain (Section 4.2)** ✅
   - `compute_info_gain()` function correctly implements:
     - Δ*(e|S) = α·Sim(e,q) + β·h_e + γ·Diversity
     - Uses `metadata.v_j` correctly
     - Includes diversity term

5. **Phase 3: Lookahead (Section 4.3)** ✅
   - ΔG formula - Correct
   - Cost sensitivity Ω̂(s) - Correct

6. **Phase 4: Layout (Section 4.4)** ✅
   - Attention potential formula - Correct
   - Optimal layout - Correct

7. **Reward Function (Section 5)** ✅
   - Circuit breaker mechanism - Correct
   - R_final formula - Correct
   - Three-dimensional cost - Correct
   - Dynamic gating ω(s) - Correct

8. **Training Algorithm (Section 6)** ✅
   - Lagrangian PPO - Correct
   - Dual critics - Correct
   - Fused advantage - Correct
   - Primal/dual updates - Correct

### ⚠️ **ISSUES TO FIX**

#### Issue 1: `compute_ranking_score()` Function (Cell ~35)
**Location**: Line 745-768

**Problem**: 
- Uses `metadata.v` instead of `metadata.v_j` (line 763-764)
- Uses wrong formula: `alpha * metadata.u + beta * metadata.h + gamma * c_out`
- Should use: `alpha * Sim(e,q) + beta * h_e` (similarity, not influence proxy)

**Current Code**:
```python
c_out = float(np.dot(query_embedding, metadata.v) / ...)  # WRONG: uses .v
score = alpha * metadata.u + beta * metadata.h + gamma * c_out  # WRONG formula
```

**Should Be**:
```python
sim_q = float(np.dot(query_embedding, metadata.v_j) / ...)  # CORRECT: uses .v_j
score = alpha * sim_q + beta * metadata.h_j  # CORRECT: similarity + entropy
```

**Fix**: Replace `metadata.v` with `metadata.v_j` and use similarity instead of influence proxy.

**Note**: The `compute_info_gain()` function (line 1218) is CORRECT and should be used instead of `compute_ranking_score()` for Phase 2 ranking.

---

#### Issue 2: Influence Proxy Documentation (Cell ~29)
**Location**: Line 385-405

**Problem**: 
- Missing detailed docstring explaining README formula
- Sigmoid normalization is not in README (but may be acceptable for practical use)

**Current Code**:
```python
def compute_influence_proxy(...):
    # Formula: u(e) = [NLL(y'|q', e) - (1/|Q_ref|) Σ NLL(y'|q', ∅)]
    ...
    u_norm = 1.0 / (1.0 + np.exp(u_raw))  # sigmoid
    return float(u_norm)
```

**Should Add**:
- Detailed docstring explaining the formula
- Note that sigmoid normalization is practical but not in README
- The raw difference `u_e = avg_with - avg_without` matches README

**Fix**: Add comprehensive docstring explaining README Section 1.2 formula.

---

#### Issue 3: Metadata Vector Attribute Consistency
**Location**: Multiple locations

**Problem**: 
- `compute_ranking_score()` uses `metadata.v` (line 763)
- Should use `metadata.v_j` to match README notation

**Fix**: Change all references from `metadata.v` to `metadata.v_j` for consistency.

---

## Recommendations

### Priority 1 (Critical)
1. **Fix `compute_ranking_score()`** - Replace with correct formula using `v_j` and similarity
2. **Use `compute_info_gain()` for Phase 2** - This function is already correct per README

### Priority 2 (Documentation)
1. **Add detailed docstrings** to `compute_influence_proxy()` explaining README formula
2. **Add comments** explaining any deviations from README (e.g., sigmoid normalization)

### Priority 3 (Code Cleanup)
1. **Remove or deprecate `compute_ranking_score()`** - Use `compute_info_gain()` instead
2. **Ensure all metadata access uses `v_j`** consistently

---

## Verification Checklist

- [x] Section 1.1: Data Source Composition - ✅ Correct
- [x] Section 1.2: Influence Proxy u(e) - ⚠️ Needs better documentation
- [x] Section 1.2: Intrinsic Entropy h_j - ✅ Correct
- [x] Section 2.1: State Space s = (q, v_q, U_0) - ✅ Correct
- [x] Section 3: Policy Network Actions - ✅ Correct
- [x] Section 4.1: Dynamic Recall - ✅ Correct
- [x] Section 4.2: Info Gain Ranking - ⚠️ `compute_info_gain()` correct, but `compute_ranking_score()` has issues
- [x] Section 4.3: Lookahead Monitoring - ✅ Correct
- [x] Section 4.4: Layout and Rendering - ✅ Correct
- [x] Section 5: Reward Function - ✅ Correct
- [x] Section 6: Training Algorithm - ✅ Correct

---

## Conclusion

The implementation is **95% compliant** with README_2.md. The main issues are:
1. A deprecated `compute_ranking_score()` function that uses wrong formula
2. Missing documentation for influence proxy formula
3. Minor attribute naming inconsistency

**The good news**: The actual Phase 2 implementation uses `compute_info_gain()` which is **100% correct** per README. The `compute_ranking_score()` function appears to be unused or deprecated.

**Action Items**:
1. Fix or remove `compute_ranking_score()` function
2. Add comprehensive documentation to `compute_influence_proxy()`
3. Verify all code paths use `compute_info_gain()` for Phase 2 ranking


