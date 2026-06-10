# S-BOX GENERATOR DOCUMENTATION

## OVERVIEW

This document describes an S-box generation methodology based on Cellular Automata (CA) evolution and Maiorana-McFarland bent function construction. The implementation combines multiple cryptographic techniques to produce bijective substitution boxes suitable for block cipher applications.

## METHODOLOGY COMPONENTS

### 1. INITIAL PATTERN GENERATION

The methodology begins with creating a structured initial pattern using Manhattan distance metric:

```
Distance Calculation: dist = |i - center_x| + |j - center_y|
Pattern Condition: (dist + offset) mod modulo_value
Cell State: 1 if condition < threshold, else 0
```

**Parameters:**
- Grid dimensions: 64×64
- Center point: (32, 32)
- Distance offset: configurable
- Modulo value: configurable
- Threshold value: configurable

**Purpose:** Generates a symmetric, deterministic initial pattern that serves as the seed for CA evolution.

### 2. CELLULAR AUTOMATA EVOLUTION

The initial pattern undergoes iterative evolution using 2D Cellular Automata with custom rules:

**Rule Structure:**
- Birth rules: Specifies neighbor counts that cause dead cells to become alive
- Survival rules: Specifies neighbor counts that keep living cells alive
- Neighborhood: Moore neighborhood (8 adjacent cells)

**Evolution Process:**
```
For each cell (i,j):
  Count living neighbors (8-connected)
  If cell is alive AND neighbors in survival_rules:
    Cell remains alive
  Else if cell is dead AND neighbors in birth_rules:
    Cell becomes alive
  Else:
    Cell becomes dead
```

**Parameters:**
- Evolution steps: configurable (typically 32)
- Birth rules: array of neighbor counts
- Survival rules: array of neighbor counts

**Output:** Sequence of evolved grids capturing CA dynamics at each step.

### 3. MAIORANA-MCFARLAND TRANSFORMATION

This cryptographic transformation constructs bent functions with optimal nonlinearity properties:

**Function Definition:**
```
Given two n-bit inputs (x, y):
1. π(y) = permutation of y
2. g(y) = arbitrary function of y
3. f(x,y) = x·π(y) ⊕ g(y)
```

**Implementation:**
```
Input: Two 16-bit binary sequences (bits1, bits2)
Process:
  x = binary_to_integer(bits1)
  y = binary_to_integer(bits2)
  π(y) = y (identity permutation)
  g(y) = y AND 0xFF (first 8 bits)
  scalar_product = popcount(x AND π(y)) mod 2
  result = scalar_product XOR g(y)
Output: 8-bit value (0-255)
```

**Purpose:** Provides cryptographically strong nonlinear transformation with provable properties.

### 4. VALUE EXTRACTION

Values are extracted from evolved CA grids using the bent function:

**Extraction Process:**
```
For each row in 64×64 grid:
  Extract row bits
  Extract corresponding column bits
  Divide into 16-bit chunks (4 chunks per row)
  For each chunk pair:
    Apply Maiorana-McFarland transformation
    Reduce result modulo 256
    Store value
```

**Parameters:**
- Chunk size: 16 bits
- Chunks per row: 4
- Total values extracted: 256
- Output format: 16×16 matrix

**Result:** Initial S-box candidate with values in range [0, 255].

### 5. BASE GRID CONSTRUCTION

Multiple CA evolution steps are combined to create the base S-box:

**Construction Algorithm:**
```
For each CA evolution step (1 to N):
  Extract 256 values using bent function
  Form 16×16 grid
  Store as ca_step_grid

For each position (i,j) in base grid:
  Sum values from all ca_step_grids at position (i,j)
  base_grid[i,j] = sum mod 256
```

**Purpose:** Combines information from multiple CA states to enhance cryptographic properties.

### 6. UNIQUENESS ENFORCEMENT

The base grid is transformed to ensure bijectivity (all values 0-255 appear exactly once):

**Analysis Phase:**
```
Identify:
- Present values: values appearing in grid
- Missing values: values from [0,255] not in grid
- Duplicate values: values appearing multiple times
- Duplicate positions: grid positions with duplicate values
```

**Correction Phase:**
```
For each duplicate position:
  If missing_values is not empty:
    Replace duplicate with value from missing_values
  Else:
    Replace using fallback: (current_value + position) mod 256
```

**Validation:** Verify final grid contains all 256 unique values exactly once.

### 7. SHIFT GRID GENERATION

Multiple shift grids are created to guide transformation operations:

**Generation Process:**
```
Number of shift grids: 8
Steps per grid: 4

For shift_grid[k]:
  start_step = k × 4
  end_step = start_step + 4
  
  For each position (i,j):
    Sum values from ca_step_grids[start_step:end_step] at (i,j)
    shift_grid[k][i,j] = sum mod 16
```

**Purpose:** Provides pseudo-random shift amounts derived from CA evolution.

### 8. SHIFT OPERATIONS

Multiple shift operations are applied sequentially to enhance diffusion:

**Operation Types:**

**a) Diagonal Shifts:**
```
For each row i:
  shift_amount = shift_grid[i][i] mod 16
  row[i] = circular_shift_right(row[i], shift_amount)

For each column j:
  shift_amount = shift_grid[j][j] mod 16
  column[j] = circular_shift_down(column[j], shift_amount)
```

**b) Anti-Diagonal Shifts:**
```
For each row i:
  anti_diag_col = (15 - i)
  shift_amount = shift_grid[i][anti_diag_col] mod 16
  row[i] = circular_shift_right(row[i], shift_amount)

For each column j:
  anti_diag_row = (15 - j)
  shift_amount = shift_grid[anti_diag_row][j] mod 16
  column[j] = circular_shift_down(column[j], shift_amount)
```

**c) Cross Pattern Shifts:**
```
Center = 8

For each row i:
  distance = |i - center|
  cross_col = (center + distance) mod 16
  shift_amount = shift_grid[i][cross_col] mod 16
  row[i] = circular_shift_right(row[i], shift_amount)

For each column j:
  distance = |j - center|
  cross_row = (center + distance) mod 16
  shift_amount = shift_grid[cross_row][j] mod 16
  column[j] = circular_shift_down(column[j], shift_amount)
```

**Iteration:** All three shift types are applied for each of the 8 shift grids sequentially.

**Purpose:** Creates complex permutation while preserving bijectivity.

### 9. NONLINEARITY CALCULATION

The Walsh-Hadamard transform is used to compute S-box nonlinearity:

**Component Function Extraction:**
```
For each bit position b (0 to 7):
  For each input x (0 to 255):
    output = S-box[x]
    component_function[x] = (output >> b) & 1
```

**Walsh-Hadamard Transform:**
```
Convert to bipolar: f_bipolar[i] = (-1)^f[i]

Iterative WHT algorithm:
For level = 0 to log₂(256)-1:
  step = 2^level
  For j = 0 to 255 step (2×step):
    For k = 0 to step-1:
      u = wht[j+k]
      v = wht[j+k+step]
      wht[j+k] = u + v
      wht[j+k+step] = u - v
```

**Nonlinearity Computation:**
```
For each component function:
  max_absolute_wht = max(|WHT values|)
  nonlinearity = (256 - max_absolute_wht) / 2

Average nonlinearity = mean(all component nonlinearities)
Minimum nonlinearity = min(all component nonlinearities)
```

**Interpretation:** Higher nonlinearity values indicate stronger resistance to linear cryptanalysis.

## CONFIGURATION PARAMETERS

### Grid Parameters
- **Grid size:** 64×64 (4096 cells)
- **Center point:** (32, 32)
- **Value range:** [0, 255]

### Manhattan Distance Parameters
- **Offset:** Adjustable (default: 6)
- **Modulo:** Adjustable (default: 7)
- **Threshold:** Adjustable (default: 3)

### Cellular Automata Parameters
- **Birth rules:** Array of neighbor counts (e.g., [3, 4, 6])
- **Survival rules:** Array of neighbor counts (e.g., [4, 5, 6])
- **Evolution steps:** Typically 32
- **Neighborhood type:** Moore (8-connected)

### Bent Function Parameters
- **Input bits:** 16 per sequence
- **Output bits:** 8
- **Extraction count:** 256 values
- **Chunk size:** 16 bits
- **Chunks per row:** 4

### S-Box Parameters
- **Dimensions:** 16×16
- **Value range:** [0, 255]
- **Bijectivity:** Required (all values unique)

### Shift Parameters
- **Number of shift grids:** 8
- **Steps per grid:** 4
- **Shift modulo:** 16
- **Pattern types:** Diagonal, anti-diagonal, cross

## ALGORITHM WORKFLOW

```
1. Initialize Manhattan Distance Grid
   ↓
2. Evolve Grid through CA (32 steps)
   ↓
3. Extract Values using Bent Function (for each step)
   ↓
4. Construct Base Grid (combine all steps)
   ↓
5. Enforce Uniqueness (make bijective)
   ↓
6. Generate Shift Grids (8 grids)
   ↓
7. Apply Sequential Shifts (3 types × 8 grids)
   ↓
8. Verify Bijectivity
   ↓
9. Calculate Nonlinearity
   ↓
10. Output Final S-Box
```

## OUTPUT STRUCTURE

The generated S-box is a 16×16 matrix that can be stored as:

**Flattened Array (256 elements):**
```
S-box[0], S-box[1], ..., S-box[255]
where S-box[i] ∈ [0, 255]
```

**2D Matrix (16×16):**
```
Row 0:  S[0,0]   S[0,1]   ... S[0,15]
Row 1:  S[1,0]   S[1,1]   ... S[1,15]
...
Row 15: S[15,0]  S[15,1]  ... S[15,15]
```

**Indexing:**
- Input value x maps to: S-box[x] or S-box[x÷16][x mod 16]
- All values 0-255 appear exactly once (bijective property)

## VERIFICATION STEPS

### Bijectivity Check
```
unique_values = set(S-box.flatten())
is_bijective = (len(unique_values) == 256)
```

### Nonlinearity Calculation
```
For each of 8 component functions:
  Compute Walsh-Hadamard Transform
  Calculate nonlinearity from max WHT coefficient
```

### Expected Outcomes
- **Bijectivity:** Must be satisfied (256 unique values)
- **Average Nonlinearity:** Target > 100
- **Minimum Nonlinearity:** Target > 96

## IMPLEMENTATION NOTES

### Computational Complexity
- **Grid evolution:** O(N × W × H) where N = steps, W×H = grid size
- **Value extraction:** O(N × 256)
- **Uniqueness enforcement:** O(256)
- **Shift operations:** O(8 × 3 × 256)
- **Nonlinearity:** O(8 × 256 × log(256))

### Memory Requirements
- **CA grids:** ~33 grids × 4KB ≈ 132KB
- **S-box candidates:** ~32 grids × 1KB ≈ 32KB
- **Shift grids:** 8 grids × 1KB = 8KB
- **Total:** ~200KB

### Deterministic Properties
- Same parameters always produce same S-box
- No random number generation required
- Reproducible results across implementations

## CRYPTOGRAPHIC CONSIDERATIONS

### Strengths
- Deterministic generation from mathematical constructs
- Bent function provides proven nonlinearity properties
- Multiple diffusion layers through CA evolution and shifts
- Bijectivity guaranteed through correction algorithm
- Complex dependencies reduce predictability

### Design Choices
- **Manhattan distance:** Creates structured initial symmetry
- **CA evolution:** Introduces complex spatial dynamics
- **Bent function:** Ensures strong nonlinear properties
- **Multi-step combination:** Increases randomness
- **Multiple shifts:** Enhances diffusion across all positions

### Evaluation Requirements
Beyond nonlinearity, generated S-boxes should be evaluated for:
- Differential uniformity
- Linear approximation probability
- Strict avalanche criterion
- Bit independence criterion
- Algebraic complexity

## PARAMETER TUNING

### Initial Pattern
- **Higher offset:** More initial living cells
- **Lower modulo:** Coarser pattern structure
- **Higher threshold:** Fewer initial living cells

### CA Rules
- **More birth rules:** Faster growth patterns
- **More survival rules:** More stable structures
- **More evolution steps:** Greater state divergence

### Shift Operations
- **More shift grids:** Increased permutation complexity
- **Different shift modulo:** Varied shift magnitudes
- **Additional shift patterns:** Enhanced diffusion

## USAGE EXAMPLE

```python
# Initialize generator
generator = SBoxGenerator()

# Generate S-box with default parameters
result = generator.generate_sbox()

# Access generated S-box
sbox = result['sbox']  # 16×16 numpy array

# Check properties
is_bijective = result['is_unique']
avg_nonlinearity = result['nonlinearity_result']['avg_nonlinearity']
min_nonlinearity = result['nonlinearity_result']['min_nonlinearity']

# Use S-box for encryption
plaintext_byte = 0x42
ciphertext_byte = sbox.flatten()[plaintext_byte]
```

## CONCLUSION

This methodology combines cellular automata dynamics with cryptographic bent function construction to generate S-boxes with provable nonlinearity properties. The multi-stage process ensures bijectivity while introducing complex transformations that enhance cryptographic strength. The deterministic nature allows reproducible generation while the configurable parameters enable exploration of the design space.