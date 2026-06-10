#!/usr/bin/env python3
"""
Comprehensive analysis of S-box cryptanalysis and image encryption performance
"""

import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import data
from scipy.stats import pearsonr
from typing import List, Dict, Tuple
import json
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use('Agg')

class IntegratedSBoxAnalyzer:

    def __init__(self, sbox: List[int], custom_name: str = "Custom S-box"):

        self.sbox = np.array(sbox)
        self.name = custom_name
        self.n = len(sbox)
        self.input_bits = int(np.log2(self.n))
        self.output_bits = self.input_bits

        # Validate S-box
        self._validate_sbox()

    def _validate_sbox(self):
        """Validate S-box properties"""
        if self.n not in [16, 256]:
            raise ValueError(f"S-box size {self.n} not supported for image encryption. Use 16 (4x4) or 256 (16x16)")

        if not all(0 <= x < self.n for x in self.sbox):
            raise ValueError(f"S-box contains invalid values. All values must be 0 to {self.n - 1}")

    # ==================== S-BOX CRYPTANALYSIS METHODS ====================

    def is_bijective(self) -> bool:
        """Check if S-box is bijective"""
        return len(set(self.sbox)) == len(self.sbox)

    def differential_uniformity(self) -> int:
        """Calculate differential uniformity"""
        max_count = 0
        for a in range(1, self.n):
            for b in range(self.n):
                count = 0
                for x in range(self.n):
                    if self.sbox[x] ^ self.sbox[x ^ a] == b:
                        count += 1
                max_count = max(max_count, count)
        return max_count

    def walsh_transform(self, truth_table: List[int]) -> List[int]:
        """Calculate Walsh spectrum"""
        n = len(truth_table)
        walsh_values = []
        for a in range(n):
            walsh_sum = 0
            for x in range(n):
                input_dot = bin(a & x).count('1') % 2
                walsh_sum += (-1) ** (input_dot ^ truth_table[x])
            walsh_values.append(walsh_sum)
        return walsh_values

    def component_nonlinearities(self) -> List[int]:
        """Calculate nonlinearity for each output bit"""
        nonlinearities = []
        for bit_pos in range(self.output_bits):
            truth_table = [(self.sbox[x] >> bit_pos) & 1 for x in range(self.n)]
            walsh_spectrum = self.walsh_transform(truth_table)
            max_walsh = max(abs(val) for val in walsh_spectrum)
            component_nl = (self.n // 2) - (max_walsh // 2)
            nonlinearities.append(int(component_nl))
        return nonlinearities

    def vbf_nonlinearity(self) -> int:
        """Calculate Vectorial Boolean Function nonlinearity"""
        max_walsh = 0
        for a in range(1, self.n):
            for b in range(1, self.n):
                total = 0
                for x in range(self.n):
                    ax = bin(a & x).count('1') % 2
                    bx = bin(b & self.sbox[x]).count('1') % 2
                    total += (-1) ** (ax ^ bx)
                max_walsh = max(max_walsh, abs(total))
        return (self.n // 2) - max_walsh // 2

    def algebraic_degree(self) -> int:
        """Calculate algebraic degree using ANF"""
        max_degree = 0
        for bit_pos in range(self.output_bits):
            truth_table = [(self.sbox[x] >> bit_pos) & 1 for x in range(self.n)]
            anf = truth_table.copy()

            for i in range(self.input_bits):
                for mask in range(self.n):
                    if mask & (1 << i):
                        for j in range(self.n):
                            if (j & (1 << i)) == 0:
                                anf[j ^ (1 << i)] ^= anf[j]

            for term in range(1, self.n):
                if anf[term] == 1:
                    degree = bin(term).count('1')
                    max_degree = max(max_degree, degree)
        return max_degree

    def strict_avalanche_criterion(self) -> Tuple[List[List[float]], float, float]:
        """Calculate Strict Avalanche Criterion"""
        sac_matrix = []
        all_deviations = []

        for input_bit in range(self.input_bits):
            row = []
            for output_bit in range(self.output_bits):
                changed_count = 0
                for x in range(self.n):
                    x_flipped = x ^ (1 << input_bit)
                    original_bit = (self.sbox[x] >> output_bit) & 1
                    flipped_bit = (self.sbox[x_flipped] >> output_bit) & 1
                    if original_bit != flipped_bit:
                        changed_count += 1

                sac_value = changed_count / self.n
                deviation = abs(sac_value - 0.5)
                row.append(sac_value)
                all_deviations.append(deviation)
            sac_matrix.append(row)

        avg_deviation = sum(all_deviations) / len(all_deviations)
        max_deviation = max(all_deviations)
        return sac_matrix, avg_deviation, max_deviation

    def linear_approximation_probability(self) -> Tuple[float, float, int, float]:
        """Calculate Linear Approximation Probability metrics"""
        lat = np.zeros((self.n, self.n), dtype=int)

        for input_mask in range(self.n):
            for output_mask in range(self.n):
                count = 0
                for x in range(self.n):
                    input_parity = bin(input_mask & x).count('1') % 2
                    output_parity = bin(output_mask & self.sbox[x]).count('1') % 2
                    if input_parity == output_parity:
                        count += 1
                lat[input_mask][output_mask] = count - self.n // 2

        max_bias = 0
        bias_values = []
        for i in range(self.n):
            for j in range(self.n):
                if i != 0 or j != 0:
                    bias = abs(lat[i][j])
                    bias_values.append(bias)
                    max_bias = max(max_bias, bias)

        max_lap = (max_bias / self.n) if max_bias > 0 else 0
        avg_bias = sum(bias_values) / len(bias_values) if bias_values else 0
        avg_lap = (avg_bias / self.n)

        return max_lap, avg_lap, max_bias, avg_bias

    def autocorrelation_test(self) -> int:
        """Calculate maximum autocorrelation value"""
        max_autocorr = 0
        for a in range(1, self.n):
            for b in range(self.n):
                total = 0
                for x in range(self.n):
                    f_x = self.sbox[x]
                    f_x_a = self.sbox[x ^ a]
                    total += (-1) ** (bin(f_x ^ f_x_a).count('1') % 2)
                max_autocorr = max(max_autocorr, abs(total))
        return max_autocorr

    def bic_nonlinearity(self) -> Dict[str, float]:
        """Calculate Bit Independence Criterion nonlinearity"""
        bic_nls = []
        for i in range(self.output_bits):
            for j in range(i + 1, self.output_bits):
                # Calculate nonlinearity between output bits i and j
                truth_table_i = [(self.sbox[x] >> i) & 1 for x in range(self.n)]
                truth_table_j = [(self.sbox[x] >> j) & 1 for x in range(self.n)]

                # XOR of two output bits
                xor_truth_table = [truth_table_i[x] ^ truth_table_j[x] for x in range(self.n)]

                walsh_spectrum = self.walsh_transform(xor_truth_table)
                max_walsh = max(abs(val) for val in walsh_spectrum)
                bic_nl = (self.n // 2) - (max_walsh // 2)
                bic_nls.append(bic_nl)

        return {
            'min': min(bic_nls) if bic_nls else 0,
            'max': max(bic_nls) if bic_nls else 0,
            'mean': sum(bic_nls) / len(bic_nls) if bic_nls else 0
        }

    # ==================== IMAGE ENCRYPTION METHODS ====================

    def encrypt_image_sbox(self, image):
        """Encrypts image using S-box"""
        encrypted = np.zeros_like(image)
        flat_image = image.flatten()

        for i in range(len(flat_image)):
            encrypted.flat[i] = self.sbox[flat_image[i]]

        return encrypted

    def decrypt_image_sbox(self, encrypted):
        """Decrypts image using S-box"""
        inverse_sbox = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            inverse_sbox[self.sbox[i]] = i

        decrypted = np.zeros_like(encrypted)
        flat_encrypted = encrypted.flatten()

        for i in range(len(flat_encrypted)):
            decrypted.flat[i] = inverse_sbox[flat_encrypted[i]]

        return decrypted

    def calculate_mse(self, original, processed):
        """Calculate Mean Squared Error"""
        return np.mean((original.astype(float) - processed.astype(float)) ** 2)

    def calculate_psnr(self, original, processed):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = self.calculate_mse(original, processed)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0 if self.n == 256 else self.n - 1
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    def calculate_mae(self, original, processed):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(original.astype(float) - processed.astype(float)))

    def calculate_npcr(self, original, encrypted):
        """Calculate Number of Pixels Change Rate"""
        diff_pixels = np.sum(original != encrypted)
        total_pixels = original.size
        return (diff_pixels / total_pixels) * 100

    def calculate_uaci(self, original, encrypted):
        """Calculate Unified Average Changing Intensity"""
        diff = np.abs(original.astype(float) - encrypted.astype(float))
        max_val = 255 if self.n == 256 else self.n - 1
        return (np.sum(diff) / (original.size * max_val)) * 100

    def calculate_correlation_coefficient(self, image):
        """Calculate horizontal, vertical and diagonal correlation coefficients"""
        height, width = image.shape

        # Horizontal correlation
        horizontal_pairs = []
        for i in range(height):
            for j in range(width - 1):
                horizontal_pairs.append([image[i, j], image[i, j + 1]])
        horizontal_pairs = np.array(horizontal_pairs)
        horizontal_corr = pearsonr(horizontal_pairs[:, 0], horizontal_pairs[:, 1])[0] if len(
            horizontal_pairs) > 0 else 0

        # Vertical correlation
        vertical_pairs = []
        for i in range(height - 1):
            for j in range(width):
                vertical_pairs.append([image[i, j], image[i + 1, j]])
        vertical_pairs = np.array(vertical_pairs)
        vertical_corr = pearsonr(vertical_pairs[:, 0], vertical_pairs[:, 1])[0] if len(vertical_pairs) > 0 else 0

        # Diagonal correlation
        diagonal_pairs = []
        for i in range(height - 1):
            for j in range(width - 1):
                diagonal_pairs.append([image[i, j], image[i + 1, j + 1]])
        diagonal_pairs = np.array(diagonal_pairs)
        diagonal_corr = pearsonr(diagonal_pairs[:, 0], diagonal_pairs[:, 1])[0] if len(diagonal_pairs) > 0 else 0

        return {
            'horizontal': horizontal_corr,
            'vertical': vertical_corr,
            'diagonal': diagonal_corr,
            'horizontal_pairs': horizontal_pairs,
            'vertical_pairs': vertical_pairs,
            'diagonal_pairs': diagonal_pairs
        }

    def simulate_occlusion_attack(self, encrypted_image, occlusion_percent=20):
        """Occlusion attack simulation"""
        attacked_image = encrypted_image.copy()
        height, width = attacked_image.shape

        mask_size = int(np.sqrt(height * width * occlusion_percent / 100))
        start_row = np.random.randint(0, max(1, height - mask_size))
        start_col = np.random.randint(0, max(1, width - mask_size))

        attacked_image[start_row:start_row + mask_size, start_col:start_col + mask_size] = 0
        return attacked_image

    def add_noise_attack(self, encrypted_image, noise_level=0.1):
        """Gaussian noise attack simulation"""
        max_val = 255 if self.n == 256 else self.n - 1
        noise = np.random.normal(0, noise_level * max_val, encrypted_image.shape)
        noisy_image = encrypted_image.astype(float) + noise
        noisy_image = np.clip(noisy_image, 0, max_val).astype(np.uint8)
        return noisy_image

    def analyze_image_comprehensive(self, image, name):
        """Comprehensive image analysis"""
        encrypted = self.encrypt_image_sbox(image)
        decrypted = self.decrypt_image_sbox(encrypted)

        # Basic metrics
        mse = self.calculate_mse(image, encrypted)
        psnr = self.calculate_psnr(image, encrypted)
        psnr_decrypt = self.calculate_psnr(image, decrypted)
        mae = self.calculate_mae(image, encrypted)
        npcr = self.calculate_npcr(image, encrypted)
        uaci = self.calculate_uaci(image, encrypted)

        # Correlation analysis
        corr_original = self.calculate_correlation_coefficient(image)
        corr_encrypted = self.calculate_correlation_coefficient(encrypted)

        # Attack simulations
        occluded = self.simulate_occlusion_attack(encrypted)
        decrypted_occluded = self.decrypt_image_sbox(occluded)
        psnr_occluded = self.calculate_psnr(image, decrypted_occluded)

        noisy = self.add_noise_attack(encrypted)
        decrypted_noisy = self.decrypt_image_sbox(noisy)
        psnr_noisy = self.calculate_psnr(image, decrypted_noisy)

        return {
            'name': name,
            'original': image,
            'encrypted': encrypted,
            'decrypted': decrypted,
            'occluded': occluded,
            'decrypted_occluded': decrypted_occluded,
            'noisy': noisy,
            'decrypted_noisy': decrypted_noisy,
            'mse': mse,
            'psnr': psnr,
            'psnr_decrypt': psnr_decrypt,
            'psnr_occluded': psnr_occluded,
            'psnr_noisy': psnr_noisy,
            'mae': mae,
            'npcr': npcr,
            'uaci': uaci,
            'corr_original': corr_original,
            'corr_encrypted': corr_encrypted
        }

    # ==================== VISUALIZATION METHODS ====================

    def create_comprehensive_metrics_visualization(self, results, save_path):
        """Metrics visualization"""
        names = [r['name'] for r in results]

        ideal_values = {
            'PSNR (Enc)': {'value': 10, 'label': 'Ideal < 10 dB'},
            'NPCR': {'value': 99.6, 'label': 'Ideal ≈ 99.6%'},
            'UACI': {'value': 33.4, 'label': 'Ideal ≈ 33.4%'},
            'MSE': {'value': 5000, 'label': 'Ideal > 5000'},
            'MAE': {'value': 70, 'label': 'Ideal > 70'},
            'Attack Rec (Occ)': {'value': 15, 'label': 'Ideal > 15 dB'},
            'Attack Rec (Noise)': {'value': 20, 'label': 'Ideal > 20 dB'}
        }

        plt.figure(figsize=(24, 14))

        metrics_data = {
            'PSNR (Enc)': [r['psnr'] for r in results],
            'NPCR': [r['npcr'] for r in results],
            'UACI': [r['uaci'] for r in results],
            'MSE': [r['mse'] for r in results],
            'MAE': [r['mae'] for r in results],
            'Attack Rec (Occ)': [r['psnr_occluded'] for r in results],
            'Attack Rec (Noise)': [r['psnr_noisy'] for r in results]
        }

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

        for i, (metric_name, values) in enumerate(metrics_data.items()):
            plt.subplot(3, 3, i + 1)
            bars = plt.bar(names, values, color=colors[i], alpha=0.8, width=0.6)

            ideal_info = ideal_values[metric_name]
            plt.axhline(y=ideal_info['value'], color='red', linestyle='--',
                        linewidth=2, alpha=0.7, label=ideal_info['label'])

            plt.title(f'{metric_name}\n{ideal_info["label"]}', fontsize=14, fontweight='bold')

            if 'PSNR' in metric_name or 'dB' in ideal_info['label']:
                plt.ylabel('dB', fontsize=12)
            elif '%' in ideal_info['label']:
                plt.ylabel('%', fontsize=12)
            else:
                plt.ylabel('Value', fontsize=12)

            plt.grid(True, alpha=0.3)

            for j, bar in enumerate(bars):
                height = bar.get_height()
                if metric_name in ['MSE', 'MAE']:
                    plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                             f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                else:
                    plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                             f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

            plt.legend(fontsize=10, loc='upper right')

            if metric_name == 'NPCR':
                plt.ylim(90, 100)
            elif metric_name == 'UACI':
                plt.ylim(20, 45)

        plt.suptitle('COMPREHENSIVE ENCRYPTION METRICS AND IDEAL VALUE COMPARISON',
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_image_process_visualization(self, results, save_path):
        """Image process visualization"""
        plt.figure(figsize=(20, 18))

        for i, result in enumerate(results):
            plt.subplot(5, 3, i + 1)
            plt.imshow(result['original'], cmap='gray')
            plt.title(f"{result['name']} - Original", fontsize=12, fontweight='bold')
            plt.axis('off')

            plt.subplot(5, 3, i + 4)
            plt.imshow(result['encrypted'], cmap='gray')
            plt.title(f"{result['name']} - Encrypted", fontsize=12, fontweight='bold')
            plt.axis('off')

            plt.subplot(5, 3, i + 7)
            plt.imshow(result['decrypted'], cmap='gray')
            plt.title(f"{result['name']} - Decrypted\nPSNR: {result['psnr_decrypt']:.2f} dB", fontsize=12,
                      fontweight='bold')
            plt.axis('off')

            plt.subplot(5, 3, i + 10)
            plt.imshow(result['decrypted_occluded'], cmap='gray')
            plt.title(f"{result['name']} - Occlusion Recovery\nPSNR: {result['psnr_occluded']:.2f} dB", fontsize=12,
                      fontweight='bold')
            plt.axis('off')

            plt.subplot(5, 3, i + 13)
            plt.hist(result['original'].flatten(), bins=min(64, self.n // 4), alpha=0.6, color='blue',
                     density=True, label='Original')
            plt.hist(result['encrypted'].flatten(), bins=min(64, self.n // 4), alpha=0.6, color='red',
                     density=True, label='Encrypted')
            plt.title(f'{result["name"]} - Histogram Comparison', fontsize=12, fontweight='bold')
            plt.xlabel('Pixel Value', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

        plt.suptitle('IMAGE ENCRYPTION PROCESS AND HISTOGRAM ANALYSIS', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_correlation_analysis(self, results, save_path):
        """Correlation analysis visualization"""
        plt.figure(figsize=(18, 12))

        for i, result in enumerate(results):
            # Original - horizontal
            plt.subplot(3, 6, i * 6 + 1)
            pairs = result['corr_original']['horizontal_pairs']
            if len(pairs) > 0:
                sample_size = min(1000, len(pairs))
                indices = np.random.choice(len(pairs), sample_size, replace=False)
                plt.scatter(pairs[indices, 0], pairs[indices, 1], alpha=0.5, s=1)
            plt.title(f'{result["name"]}\nOriginal H: {result["corr_original"]["horizontal"]:.4f}', fontsize=10)
            plt.xlabel('Pixel(x)', fontsize=9)
            plt.ylabel('Pixel(x+1)', fontsize=9)

            # Original - vertical
            plt.subplot(3, 6, i * 6 + 2)
            pairs = result['corr_original']['vertical_pairs']
            if len(pairs) > 0:
                sample_size = min(1000, len(pairs))
                indices = np.random.choice(len(pairs), sample_size, replace=False)
                plt.scatter(pairs[indices, 0], pairs[indices, 1], alpha=0.5, s=1)
            plt.title(f'Original V: {result["corr_original"]["vertical"]:.4f}', fontsize=10)
            plt.xlabel('Pixel(x,y)', fontsize=9)
            plt.ylabel('Pixel(x,y+1)', fontsize=9)

            # Original - diagonal
            plt.subplot(3, 6, i * 6 + 3)
            pairs = result['corr_original']['diagonal_pairs']
            if len(pairs) > 0:
                sample_size = min(1000, len(pairs))
                indices = np.random.choice(len(pairs), sample_size, replace=False)
                plt.scatter(pairs[indices, 0], pairs[indices, 1], alpha=0.5, s=1)
            plt.title(f'Original D: {result["corr_original"]["diagonal"]:.4f}', fontsize=10)
            plt.xlabel('Pixel(x,y)', fontsize=9)
            plt.ylabel('Pixel(x+1,y+1)', fontsize=9)

            # Encrypted - horizontal
            plt.subplot(3, 6, i * 6 + 4)
            pairs = result['corr_encrypted']['horizontal_pairs']
            if len(pairs) > 0:
                sample_size = min(1000, len(pairs))
                indices = np.random.choice(len(pairs), sample_size, replace=False)
                plt.scatter(pairs[indices, 0], pairs[indices, 1], alpha=0.5, s=1, color='red')
            plt.title(f'Encrypted H: {result["corr_encrypted"]["horizontal"]:.4f}', fontsize=10)
            plt.xlabel('Pixel(x)', fontsize=9)
            plt.ylabel('Pixel(x+1)', fontsize=9)

            # Encrypted - vertical
            plt.subplot(3, 6, i * 6 + 5)
            pairs = result['corr_encrypted']['vertical_pairs']
            if len(pairs) > 0:
                sample_size = min(1000, len(pairs))
                indices = np.random.choice(len(pairs), sample_size, replace=False)
                plt.scatter(pairs[indices, 0], pairs[indices, 1], alpha=0.5, s=1, color='red')
            plt.title(f'Encrypted V: {result["corr_encrypted"]["vertical"]:.4f}', fontsize=10)
            plt.xlabel('Pixel(x,y)', fontsize=9)
            plt.ylabel('Pixel(x,y+1)', fontsize=9)

            # Encrypted - diagonal
            plt.subplot(3, 6, i * 6 + 6)
            pairs = result['corr_encrypted']['diagonal_pairs']
            if len(pairs) > 0:
                sample_size = min(1000, len(pairs))
                indices = np.random.choice(len(pairs), sample_size, replace=False)
                plt.scatter(pairs[indices, 0], pairs[indices, 1], alpha=0.5, s=1, color='red')
            plt.title(f'Encrypted D: {result["corr_encrypted"]["diagonal"]:.4f}', fontsize=10)
            plt.xlabel('Pixel(x,y)', fontsize=9)
            plt.ylabel('Pixel(x+1,y+1)', fontsize=9)

        plt.suptitle('CORRELATION ANALYSIS - ORIGINAL vs ENCRYPTED', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_sbox_crypto_visualization(self, crypto_results, save_path):
        """S-box cryptanalysis visualization"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        fig.suptitle(f'{self.name} - Cryptographic Properties Analysis',
                     fontsize=16, fontweight='bold', y=0.95)

        # 1. LINEAR BIAS DISTRIBUTION
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_linear_bias_distribution(ax1, crypto_results)

        # 2. Component Nonlinearities
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_component_nonlinearities(ax2, crypto_results['nonlinearity']['component_values'])

        # 3. CRYPTOGRAPHIC STRENGTH RADAR CHART
        ax3 = fig.add_subplot(gs[0, 2], polar=True)
        self._plot_crypto_strength_radar(ax3, crypto_results)

        # 4. SAC Matrix Heatmap
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_sac_heatmap(ax4, crypto_results['avalanche']['sac_matrix'])

        # 5. S-BOX VALUE MATRIX
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_sbox_matrix(ax5)

        # 6. Cryptographic Strength Gauge
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_strength_gauge(ax6, crypto_results)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_linear_bias_distribution(self, ax, results):
        """Linear bias distribution plot"""
        bias_values = []
        for i in range(1, self.n):
            for j in range(1, self.n):
                count = 0
                for x in range(self.n):
                    input_parity = bin(i & x).count('1') % 2
                    output_parity = bin(j & self.sbox[x]).count('1') % 2
                    if input_parity == output_parity:
                        count += 1
                bias = abs(count - self.n // 2) / self.n
                bias_values.append(bias)

        n, bins, patches = ax.hist(bias_values, bins=30, alpha=0.7, color='skyblue',
                                   edgecolor='black', density=True)

        # Show average
        mean_bias = np.mean(bias_values)
        ax.axvline(x=mean_bias, color='red', linestyle='--', linewidth=2,
                   label=f'Mean Bias: {mean_bias:.6f}')

        ax.set_title('Linear Bias Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Bias Value')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Max: {max(bias_values):.4f}\nMin: {min(bias_values):.4f}\nStd: {np.std(bias_values):.6f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    def _plot_component_nonlinearities(self, ax, component_nls):
        """Component nonlinearities plot"""
        if len(component_nls) <= 16:
            bars = ax.bar(range(len(component_nls)), component_nls,
                          color='lightgreen', alpha=0.7, edgecolor='darkgreen')

            ax.set_title('Component Nonlinearities', fontsize=12, fontweight='bold')
            ax.set_xlabel('Output Bit Position')
            ax.set_ylabel('Nonlinearity Value')
            ax.grid(True, alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

            # Show statistics
            stats_text = f'Min: {min(component_nls)}\nMax: {max(component_nls)}\nMean: {np.mean(component_nls):.1f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    fontsize=9)
        else:
            # Simplified chart for large S-boxes
            ax.hist(component_nls, bins=20, alpha=0.7, color='lightgreen',
                    edgecolor='darkgreen')
            ax.set_title('Nonlinearity Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Nonlinearity Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

    def _plot_crypto_strength_radar(self, ax, results):
        """Cryptographic strength radar chart"""
        categories = ['Nonlinearity', 'Differential', 'Linear', 'Avalanche', 'Algebraic', 'Bit Indep.']

        # Normalized values (0-10 range)
        nl_score = min(10, results['nonlinearity']['min'] / (self.n // 4) * 10)
        diff_score = min(10, (32 - results['differential']['uniformity']) / 3.2)
        linear_score = min(10, (0.5 - results['linear_properties']['max_lap']) * 20)
        avalanche_score = min(10, (0.05 - abs(results['avalanche']['sac_average'] - 0.5)) * 200)
        algebraic_score = min(10, results['algebraic']['degree'] / self.input_bits * 10)
        bic_score = min(10, results['bic_nonlinearity']['mean'] / (self.n // 4) * 10)

        scores = [nl_score, diff_score, linear_score, avalanche_score, algebraic_score, bic_score]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]

        ax.plot(angles, scores, 'o-', linewidth=2, label='Cryptographic Strength', color='blue')
        ax.fill(angles, scores, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'])
        ax.grid(True)
        ax.set_title('Cryptographic Strength Profile', size=12, fontweight='bold', pad=20)

        # Show overall score
        overall_score = np.mean(scores[:-1])  # Don't count the last one twice
        ax.text(0.5, 0.5, f'{overall_score:.1f}/10', transform=ax.transAxes,
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.3))

    def _plot_sac_heatmap(self, ax, sac_matrix):
        """SAC matrix heatmap"""
        if len(sac_matrix) <= 16:
            sac_array = np.array(sac_matrix)
            im = ax.imshow(sac_array, cmap='coolwarm', vmin=0.4, vmax=0.6, aspect='auto')

            ax.set_title('SAC Matrix',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Output Bit')
            ax.set_ylabel('Input Bit')

            # Add number labels (for small matrices)
            if len(sac_matrix) <= 8:
                for i in range(len(sac_matrix)):
                    for j in range(len(sac_matrix[0])):
                        ax.text(j, i, f'{sac_matrix[i][j]:.3f}',
                                ha='center', va='center', fontsize=8, fontweight='bold',
                                color='black' if 0.45 < sac_matrix[i][j] < 0.55 else 'white')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Show average
            sac_flat = [item for sublist in sac_matrix for item in sublist]
            avg_sac = np.mean(sac_flat)
            ax.text(0.02, 1.02, f'Avg: {avg_sac:.4f}', transform=ax.transAxes,
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'SAC Matrix too large\nfor visualization',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('SAC Matrix (Too Large)', fontsize=12, fontweight='bold')

    def _plot_sbox_matrix(self, ax):
        """S-box value matrix"""
        if self.n <= 256:
            try:
                size = int(np.sqrt(self.n))
                sbox_2d = np.array(self.sbox).reshape(size, size)
                im = ax.imshow(sbox_2d, cmap='viridis', aspect='auto')
                ax.set_title('S-box Value Matrix', fontsize=12, fontweight='bold')
                ax.set_xlabel('Column')
                ax.set_ylabel('Row')

                # Show values for small matrices
                if size <= 16:
                    for i in range(size):
                        for j in range(size):
                            ax.text(j, i, f'{sbox_2d[i, j]}', ha='center', va='center',
                                    fontsize=6, color='white' if sbox_2d[i, j] > self.n / 2 else 'black')

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except:
                ax.text(0.5, 0.5, 'S-box matrix\ncannot be displayed',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('S-box Matrix', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'S-box too large\nfor matrix',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('S-box Matrix (Too Large)', fontsize=12, fontweight='bold')

    def _plot_strength_gauge(self, ax, results):
        """Cryptographic strength gauge"""
        # Calculate strength score
        nl_score = min(10, results['nonlinearity']['min'] / (self.n // 4) * 10)
        diff_score = min(10, (32 - results['differential']['uniformity']) / 3.2)
        linear_score = min(10, (0.5 - results['linear_properties']['max_lap']) * 20)
        avalanche_score = min(10, (0.05 - abs(results['avalanche']['sac_average'] - 0.5)) * 200)
        algebraic_score = min(10, results['algebraic']['degree'] / self.input_bits * 10)
        bic_score = min(10, results['bic_nonlinearity']['mean'] / (self.n // 4) * 10)

        overall_score = np.mean([nl_score, diff_score, linear_score, avalanche_score, algebraic_score, bic_score])

        # Convert to percentage (0-100%)
        percentage = overall_score * 10

        # Strength levels
        security_levels = ['Very Weak', 'Weak', 'Average', 'Good', 'Excellent']
        level_index = min(4, int(overall_score / 2.5))
        colors = ['#ff4757', '#ffa502', '#ffd32a', '#2ed573', '#1e90ff']

        # Large percentage in center
        ax.text(0.5, 0.6, f'{percentage:.1f}%', ha='center', va='center',
                fontsize=48, fontweight='bold', color=colors[level_index])

        # "Cryptographic Strength" text
        ax.text(0.5, 0.4, 'Cryptographic Strength', ha='center', va='center',
                fontsize=18, fontweight='bold', color='black')

        # Level
        ax.text(0.5, 0.3, security_levels[level_index], ha='center', va='center',
                fontsize=14, fontweight='bold', color=colors[level_index])

        # Background color
        ax.set_facecolor('#f0f0f0')

        # Border
        for spine in ax.spines.values():
            spine.set_color(colors[level_index])
            spine.set_linewidth(3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # ==================== REPORT GENERATION METHODS ====================

    def generate_comprehensive_report(self, crypto_results, image_results, timestamp):
        """Generate comprehensive text report"""
        # Calculate average values
        avg_psnr_enc = np.mean([r['psnr'] for r in image_results])
        avg_npcr = np.mean([r['npcr'] for r in image_results])
        avg_uaci = np.mean([r['uaci'] for r in image_results])
        avg_mae = np.mean([r['mae'] for r in image_results])
        avg_mse = np.mean([r['mse'] for r in image_results])
        avg_corr_reduction = np.mean([abs(r['corr_original']['horizontal']) - abs(r['corr_encrypted']['horizontal'])
                                      for r in image_results])
        txt_filename = f'sbox_analysis_{timestamp}.txt'

        report_content = f"""
================================================================================
                  S-BOX AND IMAGE ENCRYPTION ANALYSIS REPORT
================================================================================

Analysis Date: {timestamp}
S-box Name: {self.name}
S-box Size: {self.input_bits}x{self.output_bits} ({self.n} elements)
Test images: {', '.join([r['name'] for r in image_results])}

================================================================================
                       PART I: S-BOX CRYPTANALYSIS
================================================================================

1. BASIC S-BOX INFORMATION
-------------------------
S-box Name: {crypto_results['sbox_info']['name']}
Size: {crypto_results['sbox_info']['size']}
Number of elements: {crypto_results['sbox_info']['elements']}
Bijective S-box: {'Yes' if crypto_results['sbox_info']['bijective'] else 'No'}

2. CRYPTOGRAPHIC PROPERTIES
------------------------------
2.1. NONLINEARITY
    • Minimum nonlinearity: {crypto_results['nonlinearity']['min']}
    • Maximum nonlinearity: {crypto_results['nonlinearity']['max']}
    • Average nonlinearity: {crypto_results['nonlinearity']['mean']:.4f}
    • VBF nonlinearity: {crypto_results['nonlinearity']['vbf']}

2.2. DIFFERENTIAL PROPERTIES
    • Differential uniformity: {crypto_results['differential']['uniformity']}

2.3. AVALANCHE PROPERTIES (SAC)
    • SAC average: {crypto_results['avalanche']['sac_average']:.6f}
    • SAC deviation: {crypto_results['avalanche']['sac_deviation']:.6f}

2.4. LINEAR PROPERTIES
    • Maximum linear approximation probability (LAP): {crypto_results['linear_properties']['max_lap']:.8f}
    • Average linear approximation probability: {crypto_results['linear_properties']['avg_lap']:.8f}
    • Maximum linear bias: {crypto_results['linear_properties']['max_bias']}
    • Average linear bias: {crypto_results['linear_properties']['avg_bias']:.4f}
    • Maximum autocorrelation: {crypto_results['autocorrelation']['max_autocorr']}

2.5. BIT INDEPENDENCE (BIC-NL)
    • Minimum BIC-NL: {crypto_results['bic_nonlinearity']['min']}
    • Maximum BIC-NL: {crypto_results['bic_nonlinearity']['max']}
    • Average BIC-NL: {crypto_results['bic_nonlinearity']['mean']:.4f}

2.6. ALGEBRAIC PROPERTIES
    • Algebraic degree: {crypto_results['algebraic']['degree']}

================================================================================
                   PART II: IMAGE ENCRYPTION ANALYSIS
================================================================================

3. SUMMARY RESULTS
------------------
Average PSNR (Encryption)    : {avg_psnr_enc:.2f} dB
Average NPCR                 : {avg_npcr:.2f}%
Average UACI                 : {avg_uaci:.2f}%
Average MSE                  : {avg_mse:.0f}
Average MAE                  : {avg_mae:.2f}
Average Correlation Reduction: {avg_corr_reduction:.4f}

4. DETAILED IMAGE ANALYSIS
------------------------
"""

        # Detailed analysis for each image
        for r in image_results:
            report_content += f"""
--- {r['name']} Image ---

Performance Metrics:
  PSNR (Encryption): {r['psnr']:.2f} dB
  PSNR (Decryption): {r['psnr_decrypt']:.2f} dB
  MSE              : {r['mse']:.0f}
  MAE              : {r['mae']:.2f}

Security Metrics:
  NPCR             : {r['npcr']:.2f}%
  UACI             : {r['uaci']:.2f}%

Correlation Analysis:
  Horizontal: {r['corr_original']['horizontal']:.4f} → {r['corr_encrypted']['horizontal']:.4f}
  Vertical  : {r['corr_original']['vertical']:.4f} → {r['corr_encrypted']['vertical']:.4f}
  Diagonal  : {r['corr_original']['diagonal']:.4f} → {r['corr_encrypted']['diagonal']:.4f}

Attack Resistance:
  Occlusion Attack PSNR: {r['psnr_occluded']:.2f} dB
  Noise Attack PSNR    : {r['psnr_noisy']:.2f} dB
"""

        report_content += f"""
================================================================================
                            PART III: EVALUATION
================================================================================

5. S-BOX AND IMAGE ENCRYPTION COMPATIBILITY
--------------------------------------

5.1. S-BOX CRYPTOGRAPHIC QUALITY
{'✓ High' if crypto_results['nonlinearity']['min'] >= self.n // 4 else '⚠ Medium' if crypto_results['nonlinearity']['min'] >= self.n // 8 else '✗ Low'} Nonlinearity quality
{'✓ Good' if crypto_results['differential']['uniformity'] <= 8 else '⚠ Medium' if crypto_results['differential']['uniformity'] <= 16 else '✗ Weak'} Differential resistance
{'✓ Ideal' if abs(crypto_results['avalanche']['sac_average'] - 0.5) < 0.01 else '⚠ Acceptable' if abs(crypto_results['avalanche']['sac_average'] - 0.5) < 0.05 else '✗ Problematic'} SAC property
{'✓ Complete' if crypto_results['sbox_info']['bijective'] else '✗ Incomplete'} Bijective property

5.2. IMAGE ENCRYPTION EFFECTIVENESS
{'✓ Excellent' if avg_psnr_enc < 10 else '⚠ Good' if avg_psnr_enc < 15 else '✗ Weak'} Encryption effectiveness (PSNR: {avg_psnr_enc:.2f} dB)
{'✓ Excellent' if avg_npcr > 99 else '⚠ Good' if avg_npcr > 95 else '✗ Weak'} Pixel change ratio (NPCR: {avg_npcr:.2f}%)
{'✓ Ideal' if 30 < avg_uaci < 36 else '⚠ Acceptable' if 25 < avg_uaci < 40 else '✗ Problematic'} Intensity change uniformity (UACI: {avg_uaci:.2f}%)
{'✓ Good' if avg_corr_reduction > 0.5 else '⚠ Medium' if avg_corr_reduction > 0.2 else '✗ Weak'} Correlation weakening

6. GENERAL RECOMMENDATIONS
-------------------
"""

        # Recommendations
        recommendations = []

        if crypto_results['nonlinearity']['min'] < self.n // 4:
            recommendations.append("• S-box nonlinearity should be improved")

        if crypto_results['differential']['uniformity'] > 8:
            recommendations.append("• Differential uniformity is too high, S-box should be optimized")

        if not crypto_results['sbox_info']['bijective']:
            recommendations.append("• S-box is not bijective, not suitable for image encryption")

        if avg_psnr_enc > 15:
            recommendations.append("• Encryption effectiveness is low")

        if avg_npcr < 95:
            recommendations.append("• NPCR value should be increased")

        if not (30 < avg_uaci < 36):
            recommendations.append("• UACI value is not in ideal range")

        if not recommendations:
            report_content += "• S-box is suitable for both cryptographic and image encryption applications\n"
            report_content += "• No additional optimization required\n"
        else:
            for rec in recommendations:
                report_content += rec + "\n"

        report_content += f"""
================================================================================
                           TECHNICAL INFORMATION
================================================================================

S-box Type           : {self.name}
Test Images          : {', '.join([r['name'] for r in image_results])}
Analysis Tools       : Python, NumPy, scikit-image, matplotlib
Cryptographic Metrics: Nonlinearity, DU, SAC, LAP, BIC-NL, Algebraic Degree
Image Metrics        : PSNR, MSE, MAE, NPCR, UACI, Correlation Analysis
Visualizations       : 4 PNG files created
Attack Simulation    : Occlusion (20%) and Gaussian Noise (10%)

================================================================================
                                CONCLUSION
================================================================================

This S-box has been tested for both cryptographic properties and image encryption
performance. For detailed visual analysis, see the generated PNG files.

Analysis completed: {timestamp}
================================================================================
"""

        return report_content

    def create_comprehensive_pdf_report(self, crypto_results, image_results, txt_report, image_paths, timestamp, output_dir='.'):
        """Create comprehensive PDF report"""
        import os
        pdf_filename = os.path.join(output_dir, f"sbox_analysis_{timestamp}.pdf")

        with PdfPages(pdf_filename) as pdf:
            # Page 1: Text report
            fig = plt.figure(figsize=(11, 14))
            ax = fig.add_subplot(111)
            ax.axis('off')

            ax.text(0.05, 0.98, txt_report, transform=ax.transAxes,
                    fontsize=6, verticalalignment='top', fontfamily='monospace')

            plt.title('S-BOX AND IMAGE ENCRYPTION ANALYSIS REPORT',
                      fontsize=12, fontweight='bold', pad=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 2: S-box cryptanalysis
            img = plt.imread(image_paths['crypto'])
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            plt.title('S-BOX CRYPTOGRAPHIC PROPERTIES ANALYSIS', fontsize=16, fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 3: Image process visualization
            img = plt.imread(image_paths['process'])
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            plt.title('IMAGE PROCESS VISUALIZATION', fontsize=16, fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 4: Metrics visualization
            img = plt.imread(image_paths['metrics'])
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            plt.title('METRICS AND IDEAL VALUE COMPARISON', fontsize=16, fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 5: Correlation analysis
            img = plt.imread(image_paths['correlation'])
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            plt.title('CORRELATION ANALYSIS', fontsize=16, fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        return pdf_filename

    # ==================== MAIN ANALYSIS METHODS ====================

    def comprehensive_sbox_analysis(self) -> Dict:
        """S-box cryptanalysis"""
        print(f"🔍 S-box cryptanalysis starting: {self.name}")

        comp_nls = self.component_nonlinearities()
        min_nl = min(comp_nls)
        max_nl = max(comp_nls)
        mean_nl = sum(comp_nls) / len(comp_nls)
        vbf_nl = self.vbf_nonlinearity()
        du = self.differential_uniformity()

        sac_matrix, sac_avg_deviation, sac_max_deviation = self.strict_avalanche_criterion()
        max_lap, avg_lap, max_bias, avg_bias = self.linear_approximation_probability()
        alg_degree = self.algebraic_degree()
        is_bij = self.is_bijective()
        max_autocorr = self.autocorrelation_test()
        bic_nl = self.bic_nonlinearity()

        sac_values = []
        for row in sac_matrix:
            sac_values.extend(row)
        overall_sac = sum(sac_values) / len(sac_values)

        results = {
            'sbox_info': {
                'name': self.name,
                'size': f"{self.input_bits}x{self.output_bits}",
                'elements': self.n,
                'bijective': is_bij
            },
            'nonlinearity': {
                'min': min_nl,
                'max': max_nl,
                'mean': mean_nl,
                'vbf': vbf_nl,
                'component_values': comp_nls
            },
            'differential': {
                'uniformity': du
            },
            'avalanche': {
                'sac_average': overall_sac,
                'sac_deviation': sac_avg_deviation,
                'sac_matrix': sac_matrix
            },
            'linear_properties': {
                'max_lap': max_lap,
                'avg_lap': avg_lap,
                'max_bias': max_bias,
                'avg_bias': avg_bias
            },
            'autocorrelation': {
                'max_autocorr': max_autocorr
            },
            'bic_nonlinearity': bic_nl,
            'algebraic': {
                'degree': alg_degree
            }
        }

        print(f"✅ S-box cryptanalysis completed")
        return results

    def comprehensive_image_analysis(self):
        """Comprehensive image analysis"""
        print(f"🖼️ Image encryption analysis starting...")

        # Load test images
        try:
            lena = data.astronaut()[:, :, 0]
            image_name1 = "Astronaut"
        except:
            lena = data.camera()
            image_name1 = "Camera"

        cameraman = data.camera()
        coins = data.coins()

        # Limit image sizes according to S-box
        if self.n == 16:  # 4-bit S-box
            lena = (lena / 16).astype(np.uint8)
            cameraman = (cameraman / 16).astype(np.uint8)
            coins = (coins / 16).astype(np.uint8)

        results = []
        results.append(self.analyze_image_comprehensive(lena, image_name1))
        results.append(self.analyze_image_comprehensive(cameraman, "Cameraman"))
        results.append(self.analyze_image_comprehensive(coins, "Coins"))

        print(f"✅ 3 image analyses completed")
        return results

    def run_full_analysis(self, generate_comprehensive_reports=True, output_dir='.'):
        """Run full analysis"""
        import os
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        print("🚀 S-box and Image Encryption Analysis")
        print("=" * 80)

        # 1. S-box cryptanalysis
        crypto_results = self.comprehensive_sbox_analysis()

        # 2. Image encryption analysis
        image_results = self.comprehensive_image_analysis()

        # 3. Show console results
        self._print_console_results(crypto_results, image_results)

        if generate_comprehensive_reports:
            # 4. Create visualizations
            print(f"\n📊 Creating visualizations...")

            image_paths = {
                'crypto': os.path.join(output_dir, f'sbox_crypto_{current_time}.png'),
                'process': os.path.join(output_dir, f'image_process_{current_time}.png'),
                'metrics': os.path.join(output_dir, f'metrics_{current_time}.png'),
                'correlation': os.path.join(output_dir, f'correlation_{current_time}.png')
            }

            self.create_sbox_crypto_visualization(crypto_results, image_paths['crypto'])
            self.create_image_process_visualization(image_results, image_paths['process'])
            self.create_comprehensive_metrics_visualization(image_results, image_paths['metrics'])
            self.create_correlation_analysis(image_results, image_paths['correlation'])

            print(f"✅ 4 PNG files created")

            # 5. Create text report
            print(f"\n📝 Generating comprehensive report...")
            txt_report = self.generate_comprehensive_report(crypto_results, image_results, current_time)
            txt_filename = os.path.join(output_dir, f'sbox_analysis_{current_time}.txt')

            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(txt_report)

            print(f"✅ TXT report created: {txt_filename}")

            # 6. Create PDF report
            print(f"\n📄 Creating PDF report...")
            pdf_filename = self.create_comprehensive_pdf_report(
                crypto_results, image_results, txt_report, image_paths, current_time, output_dir)
            print(f"✅ PDF report created: {pdf_filename}")

            # 7. Save JSON data
            print(f"\n💾 Saving JSON data...")
            self._save_json_data(crypto_results, image_results, current_time, output_dir)

            # 8. Final results
            self._print_final_summary(crypto_results, image_results, current_time,
                                      txt_filename, pdf_filename, comprehensive=True)
        else:
            # Simple reports
            print(f"\n📝 Creating simple reports...")
            self._generate_simple_reports(crypto_results, image_results, current_time, output_dir)

        return crypto_results, image_results

    def _print_console_results(self, crypto_results, image_results):
        """Print console results"""
        print(f"\n=== S-BOX CRYPTOGRAPHIC PROPERTIES ===")
        print("=" * 80)
        print(f"S-box Name: {crypto_results['sbox_info']['name']}")
        print(f"Size: {crypto_results['sbox_info']['size']} ({crypto_results['sbox_info']['elements']} elements)")
        print(f"Bijective: {'✓ Yes' if crypto_results['sbox_info']['bijective'] else '✗ No'}")
        print(f"Min Nonlinearity: {crypto_results['nonlinearity']['min']}")
        print(f"VBF Nonlinearity: {crypto_results['nonlinearity']['vbf']}")
        print(f"Differential Uniformity: {crypto_results['differential']['uniformity']}")
        print(f"SAC Average: {crypto_results['avalanche']['sac_average']:.6f}")
        print(f"BIC-NL Min: {crypto_results['bic_nonlinearity']['min']}")
        print(f"Algebraic Degree: {crypto_results['algebraic']['degree']}")

        print(f"\n=== IMAGE ENCRYPTION METRICS ===")
        print("=" * 80)
        print(f"{'Image':<12} {'MSE':<8} {'MAE':<8} {'PSNR(E)':<10} {'NPCR':<8} {'UACI':<8}")
        print("=" * 80)

        for result in image_results:
            print(f"{result['name']:<12} {result['mse']:<8.0f} {result['mae']:<8.2f} "
                  f"{result['psnr']:<10.2f} {result['npcr']:<8.2f} {result['uaci']:<8.2f}")

        print("=" * 80)

    def _save_json_data(self, crypto_results, image_results, timestamp, output_dir='.'):
        """Save JSON data"""
        import os
        # Calculate average values
        avg_psnr_enc = np.mean([r['psnr'] for r in image_results])
        avg_npcr = np.mean([r['npcr'] for r in image_results])
        avg_uaci = np.mean([r['uaci'] for r in image_results])
        avg_mae = np.mean([r['mae'] for r in image_results])
        avg_mse = np.mean([r['mse'] for r in image_results])

        json_data = {
            "analysis_info": {
                "timestamp": timestamp,
                "sbox_name": self.name,
                "sbox_size": f"{self.input_bits}x{self.output_bits}",
                "images_analyzed": len(image_results),
                "analysis_type": "Integrated S-box and Image Encryption"
            },
            "sbox_cryptanalysis": {
                "bijective": crypto_results['sbox_info']['bijective'],
                "min_nonlinearity": int(crypto_results['nonlinearity']['min']),
                "vbf_nonlinearity": int(crypto_results['nonlinearity']['vbf']),
                "differential_uniformity": int(crypto_results['differential']['uniformity']),
                "sac_average": float(crypto_results['avalanche']['sac_average']),
                "algebraic_degree": int(crypto_results['algebraic']['degree']),
                "max_lap": float(crypto_results['linear_properties']['max_lap']),
                "max_autocorrelation": int(crypto_results['autocorrelation']['max_autocorr']),
                "bic_nonlinearity_min": int(crypto_results['bic_nonlinearity']['min']),
                "bic_nonlinearity_max": int(crypto_results['bic_nonlinearity']['max']),
                "bic_nonlinearity_mean": float(crypto_results['bic_nonlinearity']['mean'])
            },
            "image_encryption_summary": {
                "avg_psnr_encrypted": float(avg_psnr_enc),
                "avg_npcr": float(avg_npcr),
                "avg_uaci": float(avg_uaci),
                "avg_mse": float(avg_mse),
                "avg_mae": float(avg_mae)
            },
            "individual_image_results": []
        }

        # Detailed results for each image
        for r in image_results:
            individual_data = {
                "image_name": r['name'],
                "mse": float(r['mse']),
                "mae": float(r['mae']),
                "psnr_encrypted": float(r['psnr']),
                "psnr_decrypted": float(r['psnr_decrypt']),
                "psnr_occluded": float(r['psnr_occluded']),
                "psnr_noisy": float(r['psnr_noisy']),
                "npcr": float(r['npcr']),
                "uaci": float(r['uaci']),
                "correlation_original": {
                    "horizontal": float(r['corr_original']['horizontal']),
                    "vertical": float(r['corr_original']['vertical']),
                    "diagonal": float(r['corr_original']['diagonal'])
                },
                "correlation_encrypted": {
                    "horizontal": float(r['corr_encrypted']['horizontal']),
                    "vertical": float(r['corr_encrypted']['vertical']),
                    "diagonal": float(r['corr_encrypted']['diagonal'])
                }
            }
            json_data["individual_image_results"].append(individual_data)

        json_filename = os.path.join(output_dir, f"sbox_analysis_data_{timestamp}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON data saved: {json_filename}")

    def _generate_simple_reports(self, crypto_results, image_results, timestamp, output_dir='.'):
        """Generate simple reports"""
        import os
        # Simple TXT report
        txt_content = f"S-box Analysis Report - {timestamp}\n"
        txt_content += f"S-box: {self.name}\n"
        txt_content += f"Nonlinearity: {crypto_results['nonlinearity']['min']}\n"
        txt_content += f"Differential Uniformity: {crypto_results['differential']['uniformity']}\n\n"

        for r in image_results:
            txt_content += f"{r['name']}: PSNR={r['psnr']:.2f}dB, NPCR={r['npcr']:.2f}%, UACI={r['uaci']:.2f}%\n"

        txt_filename = os.path.join(output_dir, f'simple_analysis_{timestamp}.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        print(f"✅ Simple TXT report created: {txt_filename}")

    def _print_final_summary(self, crypto_results, image_results, timestamp, txt_filename, pdf_filename,
                             comprehensive=True):
        """Print final summary"""
        avg_psnr_enc = np.mean([r['psnr'] for r in image_results])
        avg_npcr = np.mean([r['npcr'] for r in image_results])
        avg_uaci = np.mean([r['uaci'] for r in image_results])

        if comprehensive:
            print(f"\n🎉 === ANALYSIS COMPLETED ===")
        else:
            print(f"\n🎉 === SIMPLE ANALYSIS COMPLETED ===")

        print(f"📅 Date: {timestamp}")
        print(f"🔍 S-box: {self.name} ({self.input_bits}x{self.output_bits})")
        print(f"🖼️ Test images: {len(image_results)} pieces")

        print(f"\n📊 S-box Cryptanalysis Results:")
        print(f"   Nonlinearity: {crypto_results['nonlinearity']['min']}")
        print(f"   Differential Uniformity: {crypto_results['differential']['uniformity']}")
        print(f"   Bijective: {'✓' if crypto_results['sbox_info']['bijective'] else '✗'}")
        print(f"   BIC-NL: {crypto_results['bic_nonlinearity']['min']}-{crypto_results['bic_nonlinearity']['max']}")

        print(f"\n📈 Image Encryption Results:")
        print(f"   PSNR (encryption): {avg_psnr_enc:.2f} dB")
        print(f"   NPCR: {avg_npcr:.2f}%")
        print(f"   UACI: {avg_uaci:.2f}%")

        if comprehensive:
            print(f"\n📁 Created files:")
            print(f"   📄 PDF Report: {pdf_filename}")
            print(f"   📝 TXT Report: {txt_filename}")
            print(f"   💾 JSON Data: sbox_analysis_data_{timestamp}.json")
            print(f"   🖼️ PNG Visualizations: 4 pieces")
            print(f"\n✅ Comprehensive integrated analysis successfully completed!")
        else:
            print(f"\n📁 Created files:")
            print(f"   📝 TXT Report: {txt_filename}")
            print(f"\n✅ Simple analysis successfully completed!")

# ==================== HELPER FUNCTIONS ====================

def parse_sbox_from_string(sbox_str: str) -> List[int]:
    """Parse S-box from string input"""
    cleaned = sbox_str.strip().replace('[', '').replace(']', '').replace(',', ' ')
    values = [int(x) for x in cleaned.split() if x.strip()]
    return values


def load_sbox_from_file(filename: str) -> List[int]:
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            return parse_sbox_from_string(content)
    except FileNotFoundError:
        raise ValueError(f"File {filename} not found")
    except Exception as e:
        raise ValueError(f"Error reading S-box from file: {e}")


def main():
    """Main program"""
    parser = argparse.ArgumentParser(description='S-box and Image Encryption Analysis Tool')
    parser.add_argument('--sbox', type=str, help='S-box values (comma-separated)')
    parser.add_argument('--file', type=str, help='Load S-box from file')
    parser.add_argument('--name', type=str, default='Custom S-box', help='Name for the S-box')
    parser.add_argument('--simple', action='store_true', help='Simple analysis mode (without comprehensive reports)')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for results')

    args = parser.parse_args()

    print("🔐 S-box and Image Encryption Analysis Tool")
    print("=" * 80)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        if args.file:
            # Load from file
            sbox = load_sbox_from_file(args.file)
            sbox_name = f"S-box from {args.file} file"
        elif args.sbox:
            # S-box given as parameter
            sbox = parse_sbox_from_string(args.sbox)
            sbox_name = args.name
        else:
            # Interactive mode
            print("\nOptions:")
            print("1. Analyze DEFAULT S-box (16x16)")
            print("2. Enter custom S-box")

            choice = input("Select option (1-2): ").strip()

            if choice == '1':
                # DEFAULT S-box
                sbox = [
                    210, 111, 26, 106, 166, 245, 79, 185, 158, 238, 207, 144, 85, 235, 74, 51,
                    22, 213, 36, 14, 157, 161, 190, 113, 167, 209, 155, 195, 120, 61, 42, 230,
                    60, 56, 244, 183, 39, 237, 65, 182, 46, 78, 23, 148, 93, 104, 191, 156,
                    193, 171, 222, 62, 214, 211, 255, 57, 164, 118, 30, 180, 203, 186, 64, 25,
                    233, 196, 4, 45, 247, 126, 248, 5, 253, 223, 13, 204, 86, 168, 122, 27,
                    162, 12, 232, 18, 83, 34, 215, 249, 77, 3, 135, 35, 114, 49, 228, 133,
                    20, 70, 121, 68, 239, 202, 123, 97, 71, 19, 218, 38, 127, 173, 82, 16,
                    32, 246, 160, 124, 194, 187, 147, 216, 146, 29, 175, 52, 217, 169, 50, 107,
                    1, 76, 152, 24, 88, 241, 81, 140, 53, 197, 2, 236, 178, 227, 110, 37,
                    251, 44, 212, 33, 136, 63, 129, 181, 188, 138, 198, 75, 219, 199, 95, 192,
                    96, 100, 229, 234, 149, 200, 134, 7, 66, 174, 69, 242, 159, 150, 226, 101,
                    176, 47, 240, 231, 80, 125, 224, 72, 139, 21, 58, 163, 48, 102, 112, 9,
                    132, 165, 153, 205, 99, 172, 11, 54, 117, 151, 40, 145, 142, 220, 177, 6,
                    109, 103, 90, 87, 67, 17, 10, 170, 41, 221, 254, 130, 141, 201, 243, 31,
                    94, 98, 73, 8, 184, 131, 252, 55, 84, 89, 206, 208, 105, 92, 28, 91,
                    0, 116, 250, 225, 128, 143, 15, 59, 179, 108, 115, 119, 43, 137, 189, 154
                ]
                sbox_name = "DEFAULT 16x16 S-box"
            elif choice == '2':
                print("\nSelect S-box size:")
                print("1. 4x4 (16 values)")
                print("2. 16x16 (256 values)")

                size_choice = input("Size (1-2): ").strip()
                size = 16 if size_choice == '1' else 256
                size_desc = "4x4" if size == 16 else "16x16"

                print(f"\nEnter {size} values for {size_desc} S-box:")
                user_input = input("S-box values: ").strip()
                sbox = parse_sbox_from_string(user_input)
                sbox_name = f"Custom {size_desc} S-box"


        # Override name if provided
        if args.name != 'Custom S-box':
            sbox_name = args.name

        # Ask about report choice
        # Always generate comprehensive reports
        generate_comprehensive = True

        # Start analyzer and run analysis
        analyzer = IntegratedSBoxAnalyzer(sbox, sbox_name)
        crypto_results, image_results = analyzer.run_full_analysis(
            generate_comprehensive_reports=generate_comprehensive,
            output_dir=args.output_dir
        )

    except KeyboardInterrupt:
        print("\nAnalysis stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()