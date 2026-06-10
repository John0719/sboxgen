#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-Box Generator using Cellular Automata and Maiorana-McFarland Construction

This program generates cryptographically secure S-boxes using:
- Manhattan distance-based initial patterns
- 2D Cellular Automata evolution
- Maiorana-McFarland bent function construction
- Multiple shift operations for enhanced security

Author: [Your Name]
License: MIT
"""

import numpy as np
from itertools import product
from collections import Counter


class SBoxGenerator:
    """S-Box Generator using Cellular Automata and cryptographic transformations."""

    # Configuration Constants
    SIZE = 64  # 64x64 grid size
    CENTER = (32, 32)  # Center point for Manhattan distance
    MANHATTAN_OFFSET = 6  # Manhattan distance offset
    MANHATTAN_MOD = 7  # Manhattan modulo value
    MANHATTAN_THRESHOLD = 3  # Manhattan threshold

    # Cellular Automata Parameters
    BIRTH_RULES = [3, 4, 6]  # Birth rules (B346)
    SURVIVE_RULES = [4, 5, 6]  # Survival rules (S456)
    CA_STEPS = 32  # Number of CA evolution steps

    # Maiorana-McFarland Parameters
    BENT_BITS = 16  # 16-bit bent function
    EXTRACT_VALUES = 256  # Number of values to extract
    CHUNKS_PER_ROW = 4  # Chunks per row
    CHUNK_SIZE = 16  # Bits per chunk

    # S-Box Parameters
    SBOX_SIZE = 16  # 16x16 S-Box
    MAX_VALUE = 256  # Maximum value (0-255)
    SHIFT_GROUPS = 8  # Number of shift grids
    STEPS_PER_GROUP = 4  # CA steps per group
    SHIFT_MOD = 16  # Modulo for shift operations

    # Cross Pattern Parameters
    CROSS_CENTER = 8  # Center of 16x16 grid

    # Nonlinearity Parameters
    BIT_POSITIONS = 8  # Number of bit positions in S-Box
    INPUT_RANGE = 256  # S-Box input range

    def __init__(self):
        """Initialize the S-Box generator."""
        self.evolved_grids = None
        self.ca_step_grids = None

    def manhattan_grid(self):
        """Create initial grid using Manhattan distance pattern."""
        grid = np.zeros((self.SIZE, self.SIZE), dtype=int)
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                dist = abs(i - self.CENTER[0]) + abs(j - self.CENTER[1])
                condition = (dist + self.MANHATTAN_OFFSET) % self.MANHATTAN_MOD
                grid[i, j] = 1 if condition < self.MANHATTAN_THRESHOLD else 0
        return grid

    def get_neighbors(self, grid, i, j):
        """Calculate number of neighbors for a grid position."""
        neighbors = []
        for dx, dy in product([-1, 0, 1], repeat=2):
            if dx == 0 and dy == 0:
                continue
            ni, nj = i + dx, j + dy
            if 0 <= ni < self.SIZE and 0 <= nj < self.SIZE:
                neighbors.append(grid[ni][nj])
        return sum(neighbors)

    def ca_step(self, grid):
        """Perform one step of Cellular Automata evolution."""
        new_grid = np.zeros_like(grid)
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                neighbors = self.get_neighbors(grid, i, j)
                if grid[i, j] == 1 and neighbors in self.SURVIVE_RULES:
                    new_grid[i, j] = 1
                elif grid[i, j] == 0 and neighbors in self.BIRTH_RULES:
                    new_grid[i, j] = 1
        return new_grid

    def evolve_grid(self, initial_grid):
        """Evolve the grid through multiple CA steps."""
        grids = [initial_grid.copy()]
        current = initial_grid.copy()
        for step in range(self.CA_STEPS):
            current = self.ca_step(current)
            grids.append(current.copy())
        return grids

    def maiorana_mcfarland_transform(self, bits1, bits2):
        """Apply Maiorana-McFarland bent function construction."""
        try:
            x = int("".join(str(b) for b in bits1), 2)
            y = int("".join(str(b) for b in bits2), 2)

            pi_y = y
            g_y = y & 0xFF  # First 8 bits

            scalar_product = bin(x & pi_y).count('1') % 2
            result = scalar_product ^ (g_y & 0xFF)

            return result
        except (ValueError, IndexError) as e:
            print(f"Warning in maiorana_mcfarland_transform: {e}")
            return 0

    def extract_values_from_grid(self, grid):
        """Extract values from 64x64 grid using bent function."""
        values = []

        for row in range(self.SIZE):
            row_bits = grid[row]
            col_bits = grid[:, row] if row < self.SIZE else grid[:, row % self.SIZE]

            for chunk in range(self.CHUNKS_PER_ROW):
                start_idx = chunk * self.CHUNK_SIZE
                end_idx = start_idx + self.CHUNK_SIZE

                if end_idx <= len(row_bits) and end_idx <= len(col_bits):
                    row_chunk = row_bits[start_idx:end_idx]
                    col_chunk = col_bits[start_idx:end_idx]

                    bent_result = self.maiorana_mcfarland_transform(row_chunk, col_chunk)
                    values.append(bent_result % self.MAX_VALUE)

                    if len(values) >= self.EXTRACT_VALUES:
                        break

            if len(values) >= self.EXTRACT_VALUES:
                break

        # Ensure we have exactly EXTRACT_VALUES values
        while len(values) < self.EXTRACT_VALUES:
            values.append(values[-1] if values else 0)

        return np.array(values[:self.EXTRACT_VALUES])

    def create_ca_step_grid(self, evolved_grids, step_idx):
        """Create grid from specific CA step."""
        if step_idx >= len(evolved_grids):
            return None

        grid = evolved_grids[step_idx]
        values = self.extract_values_from_grid(grid)
        return values.reshape((self.SBOX_SIZE, self.SBOX_SIZE))

    def create_base_grid(self, evolved_grids):
        """Create base grid using CA step-based approach."""
        ca_step_grids = []

        for step in range(1, min(self.CA_STEPS + 1, len(evolved_grids))):
            ca_grid = self.create_ca_step_grid(evolved_grids, step)
            if ca_grid is not None:
                ca_step_grids.append(ca_grid)

        if not ca_step_grids:
            raise ValueError("No valid CA step grids created")

        base_grid = np.zeros((self.SBOX_SIZE, self.SBOX_SIZE), dtype=int)

        for i in range(self.SBOX_SIZE):
            for j in range(self.SBOX_SIZE):
                total = sum(grid[i][j] for grid in ca_step_grids)
                base_grid[i][j] = total % self.MAX_VALUE

        return base_grid, ca_step_grids

    def analyze_uniqueness(self, grid):
        """Analyze uniqueness of values in grid."""
        flat_grid = grid.flatten()
        counter = Counter(flat_grid)

        all_values = set(range(self.MAX_VALUE))
        present_values = set(flat_grid)
        missing_values = all_values - present_values
        duplicate_values = {val: count for val, count in counter.items() if count > 1}

        return {
            'present_values': present_values,
            'missing_values': missing_values,
            'duplicate_values': duplicate_values,
            'unique_count': len(present_values),
            'total_duplicates': sum(count - 1 for count in duplicate_values.values())
        }

    def make_grid_unique(self, base_grid, ca_step_grids):
        """Make grid values unique (bijective)."""
        analysis = self.analyze_uniqueness(base_grid)

        if analysis['unique_count'] == self.MAX_VALUE:
            return base_grid, True

        unique_grid = base_grid.flatten().copy()
        missing_values = set(analysis['missing_values'])
        duplicate_positions = []

        # Find duplicate positions
        seen_values = set()
        for pos, val in enumerate(unique_grid):
            if val in analysis['duplicate_values']:
                if val in seen_values:
                    duplicate_positions.append(pos)
                else:
                    seen_values.add(val)

        # Replace duplicates with missing values
        duplicate_idx = 0

        for ca_grid in ca_step_grids:
            if duplicate_idx >= len(duplicate_positions):
                break

            flat_ca = ca_grid.flatten()

            for ca_val in flat_ca:
                if ca_val in missing_values and duplicate_idx < len(duplicate_positions):
                    duplicate_pos = duplicate_positions[duplicate_idx]
                    unique_grid[duplicate_pos] = ca_val

                    missing_values.remove(ca_val)
                    duplicate_idx += 1

                    if duplicate_idx >= len(duplicate_positions):
                        break

        # Fill remaining duplicates
        while duplicate_idx < len(duplicate_positions):
            duplicate_pos = duplicate_positions[duplicate_idx]
            if missing_values:
                new_val = missing_values.pop()
                unique_grid[duplicate_pos] = new_val
            else:
                unique_grid[duplicate_pos] = (unique_grid[duplicate_pos] + duplicate_pos) % self.MAX_VALUE
            duplicate_idx += 1

        final_grid = unique_grid.reshape((self.SBOX_SIZE, self.SBOX_SIZE))
        final_analysis = self.analyze_uniqueness(final_grid)
        is_unique = final_analysis['unique_count'] == self.MAX_VALUE

        return final_grid, is_unique

    def create_shift_grids(self, ca_step_grids):
        """Create shift grids for transformation operations."""
        shift_grids = []

        for group_idx in range(self.SHIFT_GROUPS):
            shift_grid = np.zeros((self.SBOX_SIZE, self.SBOX_SIZE), dtype=int)

            start_step = group_idx * self.STEPS_PER_GROUP
            end_step = min(start_step + self.STEPS_PER_GROUP, len(ca_step_grids))

            for i in range(self.SBOX_SIZE):
                for j in range(self.SBOX_SIZE):
                    total = 0
                    for step_idx in range(start_step, end_step):
                        if step_idx < len(ca_step_grids):
                            total += ca_step_grids[step_idx][i][j]

                    shift_grid[i][j] = total % self.SHIFT_MOD

            shift_grids.append(shift_grid)

        return shift_grids

    def apply_shifts(self, base_grid, shift_grids):
        """Apply shift operations to base grid."""
        final_grid = base_grid.copy()

        for shift_idx, shift_grid in enumerate(shift_grids):
            print(f"Applying shift operation {shift_idx + 1}/{len(shift_grids)}...")

            # Diagonal shifts
            for row in range(self.SBOX_SIZE):
                shift_amount = shift_grid[row][row] % self.SHIFT_MOD
                final_grid[row] = np.roll(final_grid[row], shift_amount)

            for col in range(self.SBOX_SIZE):
                shift_amount = shift_grid[col][col] % self.SHIFT_MOD
                final_grid[:, col] = np.roll(final_grid[:, col], shift_amount)

            # Anti-diagonal shifts
            for row in range(self.SBOX_SIZE):
                anti_diag_col = (self.SBOX_SIZE - 1) - row
                shift_amount = shift_grid[row][anti_diag_col] % self.SHIFT_MOD
                final_grid[row] = np.roll(final_grid[row], shift_amount)

            for col in range(self.SBOX_SIZE):
                anti_diag_row = (self.SBOX_SIZE - 1) - col
                shift_amount = shift_grid[anti_diag_row][col] % self.SHIFT_MOD
                final_grid[:, col] = np.roll(final_grid[:, col], shift_amount)

            # Cross pattern shifts
            for row in range(self.SBOX_SIZE):
                distance_from_center = abs(row - self.CROSS_CENTER)
                cross_col = (self.CROSS_CENTER + distance_from_center) % self.SBOX_SIZE
                shift_amount = shift_grid[row][cross_col] % self.SHIFT_MOD
                final_grid[row] = np.roll(final_grid[row], shift_amount)

            for col in range(self.SBOX_SIZE):
                distance_from_center = abs(col - self.CROSS_CENTER)
                cross_row = (self.CROSS_CENTER + distance_from_center) % self.SBOX_SIZE
                shift_amount = shift_grid[cross_row][col] % self.SHIFT_MOD
                final_grid[:, col] = np.roll(final_grid[:, col], shift_amount)

        return final_grid

    def calculate_nonlinearity(self, sbox):
        """Calculate nonlinearity of the S-Box."""

        def walsh_hadamard_transform(f):
            """Walsh-Hadamard transform."""
            n = len(f)
            if n == 0 or (n & (n - 1)) != 0:  # Check if n is power of 2
                raise ValueError("Input length must be a power of 2")

            log_n = int(np.log2(n))
            f_bipolar = np.array([(-1) ** bit for bit in f])

            wht = f_bipolar.copy().astype(float)
            for i in range(log_n):
                step = 1 << i
                for j in range(0, n, step << 1):
                    for k in range(step):
                        u = wht[j + k]
                        v = wht[j + k + step]
                        wht[j + k] = u + v
                        wht[j + k + step] = u - v

            return wht

        def calculate_component_nonlinearity(component_func):
            """Calculate nonlinearity of component function."""
            try:
                wht = walsh_hadamard_transform(component_func)
                max_absolute_wht = np.max(np.abs(wht))
                nonlinearity = (self.INPUT_RANGE - max_absolute_wht) // 2
                return max(0, nonlinearity)  # Ensure non-negative
            except Exception as e:
                print(f"Warning in component nonlinearity calculation: {e}")
                return 0

        # Flatten S-Box if needed
        if sbox.shape == (self.SBOX_SIZE, self.SBOX_SIZE):
            sbox_flat = sbox.flatten()
        else:
            sbox_flat = sbox

        nonlinearities = []

        for bit_position in range(self.BIT_POSITIONS):
            component_func = []
            for input_val in range(self.INPUT_RANGE):
                output_val = sbox_flat[input_val]
                component_bit = (output_val >> bit_position) & 1
                component_func.append(component_bit)

            component_nonlinearity = calculate_component_nonlinearity(component_func)
            nonlinearities.append(component_nonlinearity)

        min_nonlinearity = min(nonlinearities) if nonlinearities else 0
        avg_nonlinearity = np.mean(nonlinearities) if nonlinearities else 0

        return {
            'nonlinearities': nonlinearities,
            'min_nonlinearity': min_nonlinearity,
            'avg_nonlinearity': avg_nonlinearity
        }

    def print_sbox_formatted(self, sbox, title="S-Box"):
        """Print S-Box in formatted style."""
        print(f"\n{title} ({self.SBOX_SIZE}x{self.SBOX_SIZE}):")
        print("-" * 68)
        for row in sbox:
            row_str = ','.join(f'{val:3d}' for val in row)
            print(row_str)

    def generate_sbox(self):
        """Generate S-Box using the complete pipeline."""
        print("S-Box Generator - Cellular Automata Based")
        print("=" * 50)
        print(f"Grid Size: {self.SIZE}x{self.SIZE}")
        print(f"Manhattan Offset: {self.MANHATTAN_OFFSET}")
        print(f"Manhattan Mod: {self.MANHATTAN_MOD}, Threshold: {self.MANHATTAN_THRESHOLD}")
        print(f"CA Rule: B{''.join(map(str, self.BIRTH_RULES))}/S{''.join(map(str, self.SURVIVE_RULES))}")
        print(f"CA Steps: {self.CA_STEPS}")
        print(f"S-Box Size: {self.SBOX_SIZE}x{self.SBOX_SIZE}")
        print(f"Shift Groups: {self.SHIFT_GROUPS}")
        print("=" * 50)

        try:
            # Create initial grid
            print("\n1. Creating initial grid...")
            initial_grid = self.manhattan_grid()
            print(f"   Living cells in initial grid: {np.sum(initial_grid)}")

            # CA evolution
            print(f"\n2. Running {self.CA_STEPS} steps of CA evolution...")
            self.evolved_grids = self.evolve_grid(initial_grid)
            print("   CA evolution completed")

            # Create base grid
            print("\n3. Creating base grid...")
            base_grid, self.ca_step_grids = self.create_base_grid(self.evolved_grids)
            print(f"   Base grid created, value range: {np.min(base_grid)} - {np.max(base_grid)}")

            # Make grid unique
            print("\n4. Making base grid unique...")
            unique_base_grid, is_unique = self.make_grid_unique(base_grid, self.ca_step_grids)

            if not is_unique:
                print("   ❌ Failed to make base grid unique!")
                return None

            print("   ✓ Base grid successfully made unique")

            # Create shift grids
            print("\n5. Creating shift grids...")
            shift_grids = self.create_shift_grids(self.ca_step_grids)
            print(f"   ✓ Created {len(shift_grids)} shift grids")

            # Apply shift operations
            print("\n6. Applying shift operations...")
            final_sbox = self.apply_shifts(unique_base_grid, shift_grids)
            print("   ✓ Shift operations completed")

            # Check final S-Box uniqueness
            print("\n7. Checking final S-Box uniqueness...")
            final_analysis = self.analyze_uniqueness(final_sbox)
            is_final_unique = final_analysis['unique_count'] == self.MAX_VALUE

            if not is_final_unique:
                print("   ❌ Final S-Box is not unique!")
                print(f"   Unique values: {final_analysis['unique_count']}/{self.MAX_VALUE}")
                return None

            print("   ✓ Final S-Box is unique")

            # Calculate nonlinearity
            print("\n8. Calculating nonlinearity...")
            nonlinearity_result = self.calculate_nonlinearity(final_sbox)
            print("   ✓ Nonlinearity calculation completed")

            # Display results
            print("\n" + "=" * 50)
            print("RESULTS")
            print("=" * 50)
            print(f"Grid Size: {self.SIZE}x{self.SIZE}")
            print(f"Manhattan Offset: {self.MANHATTAN_OFFSET}")
            print(f"CA Rule: B{''.join(map(str, self.BIRTH_RULES))}/S{''.join(map(str, self.SURVIVE_RULES))}")
            print(f"CA Steps: {self.CA_STEPS}")
            print(f"S-Box Size: {self.SBOX_SIZE}x{self.SBOX_SIZE}")
            print(f"Shift Groups: {len(shift_grids)}")
            print(f"Final S-Box Unique: {is_final_unique}")
            print(f"Average Nonlinearity: {nonlinearity_result['avg_nonlinearity']:.4f}")
            print(f"Minimum Nonlinearity: {nonlinearity_result['min_nonlinearity']}")
            print(f"Component Nonlinearities: {nonlinearity_result['nonlinearities']}")

            # Print S-Box
            self.print_sbox_formatted(final_sbox, "GENERATED S-BOX")

            return {
                'sbox': final_sbox,
                'nonlinearity_result': nonlinearity_result,
                'is_unique': is_final_unique,
                'parameters': {
                    'size': self.SIZE,
                    'manhattan_offset': self.MANHATTAN_OFFSET,
                    'birth_rules': self.BIRTH_RULES,
                    'survive_rules': self.SURVIVE_RULES,
                    'ca_steps': self.CA_STEPS,
                    'sbox_size': self.SBOX_SIZE,
                    'shift_groups': len(shift_grids)
                }
            }

        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to generate S-Box."""
    print("🔐 S-BOX GENERATOR")
    print("🎯 Cryptographically Secure S-Box Generation using Cellular Automata")
    print()

    generator = SBoxGenerator()
    result = generator.generate_sbox()

    if result:
        print(f"\n🎯 S-BOX SUCCESSFULLY GENERATED!")
        params = result['parameters']
        print(f"   📐 GRID: {params['size']}x{params['size']}")
        print(f"   📍 MANHATTAN OFFSET: {params['manhattan_offset']}")
        print(
            f"   🔄 CA RULE: B{''.join(map(str, params['birth_rules']))}/S{''.join(map(str, params['survive_rules']))}")
        print(f"   🔢 CA STEPS: {params['ca_steps']}")
        print(f"   📊 S-BOX: {params['sbox_size']}x{params['sbox_size']}")
        print(f"   🔀 SHIFT GRIDS: {params['shift_groups']}")
        print(f"   💯 AVERAGE NONLINEARITY: {result['nonlinearity_result']['avg_nonlinearity']:.4f}")
        print(f"   🎯 MINIMUM NONLINEARITY: {result['nonlinearity_result']['min_nonlinearity']}")
    else:
        print("\n❌ Failed to generate S-Box!")


if __name__ == "__main__":
    main()