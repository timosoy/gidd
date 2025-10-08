import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InferenceStepsAnalyzer:
    def __init__(self, samples_dir: str = "Samples"):
        self.samples_dir = Path(samples_dir)
        self.metrics_data = {}
        
    def load_inference_steps_metrics(self):
        """Load metrics files for different inference steps (32, 64, 128, 256)"""
        print(f"Loading inference steps metrics from {self.samples_dir}...")
        
        # Define the inference steps we want to compare
        inference_steps = [32, 64, 128, 256]
        
        for steps in inference_steps:
            # Look for corrected samples metrics files
            pattern = f"*corrected_samples_metrics_{steps}_steps.json"
            metrics_files = list(self.samples_dir.glob(pattern))
            
            if metrics_files:
                file_path = metrics_files[0]  # Take the first match
                print(f"Found {file_path.name} for {steps} steps")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    self.metrics_data[steps] = {
                        'data': data,
                        'file_path': file_path
                    }
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
            else:
                # Fallback: map 128 steps to the "original method" metrics file if present
                if steps == 128:
                    fallback = self.samples_dir / "corrected_samples_metrics.json"
                    if fallback.exists():
                        print(f"No metrics file found for {steps} steps; using original method: {fallback.name}")
                        try:
                            with open(fallback, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            self.metrics_data[steps] = {
                                'data': data,
                                'file_path': fallback
                            }
                        except Exception as e:
                            print(f"Error loading {fallback.name}: {e}")
                    else:
                        print(f"No metrics file found for {steps} steps")
                else:
                    print(f"No metrics file found for {steps} steps")
                
        print(f"Successfully loaded {len(self.metrics_data)} inference steps metrics")
        
    def plot_perplexity_comparison(self, save_path: str = "plots/inference_steps_perplexity.png"):
        """Plot perplexity comparison across different inference steps"""
        # Prepare data
        steps_data = []
        for steps, info in self.metrics_data.items():
            data = info['data']
            
            # Extract metrics
            external_ppl = data.get('external_metrics', {}).get('ppl', None)
            self_ppl = data.get('self_ppl_metrics', {}).get('average_perplexity', None)
            
            if external_ppl is not None:
                steps_data.append({
                    'steps': steps,
                    'external_ppl': external_ppl,
                    'self_ppl': self_ppl
                })
        
        if not steps_data:
            print("No perplexity data found for plotting")
            return
            
        df = pd.DataFrame(steps_data)
        df = df.sort_values('steps')
        
        # Create separate plots for each metric (self-surprisal removed)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: External PPL vs Inference Steps
        axes[0].plot(df['steps'], df['external_ppl'], marker='o', linewidth=3, markersize=10, color='blue')
        axes[0].set_title('External PPL vs Inference Steps', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Inference Steps', fontsize=12)
        axes[0].set_ylabel('External PPL', fontsize=12)
        axes[0].set_xticks(df['steps'])
        axes[0].set_xticklabels([f'{int(x)}' for x in df['steps']])
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Self-PPL vs Inference Steps
        if df['self_ppl'].notna().any():
            axes[1].plot(df['steps'], df['self_ppl'], marker='s', linewidth=3, markersize=10, color='orange')
            axes[1].set_title('Self-PPL vs Inference Steps', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Inference Steps', fontsize=12)
            axes[1].set_ylabel('Self-PPL', fontsize=12)
            axes[1].set_xticks(df['steps'])
            axes[1].set_xticklabels([f'{int(x)}' for x in df['steps']])
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Perplexity comparison plot saved to {save_path}")
        
    def plot_improvement_metrics(self, save_path: str = "plots/inference_steps_improvement.png"):
        """Plot improvement metrics across different inference steps"""
        # Prepare data
        steps_data = []
        for steps, info in self.metrics_data.items():
            data = info['data']
            
            # Extract improvement metrics
            ppl_improvement = data.get('self_ppl_improvement', {}).get('absolute_improvement', None)
            ppl_ratio = data.get('self_ppl_improvement', {}).get('improvement_ratio', None)
            self_accuracy = data.get('average_self_accuracy', None)
            
            if ppl_improvement is not None:
                steps_data.append({
                    'steps': steps,
                    'ppl_improvement': ppl_improvement,
                    'ppl_ratio': ppl_ratio,
                    'self_accuracy': self_accuracy
                })
        
        if not steps_data:
            print("No improvement metrics data found for plotting")
            return
            
        df = pd.DataFrame(steps_data)
        df = df.sort_values('steps')
        
        # Create separate plots for each metric (remove surprisal plots)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: PPL improvement vs Inference Steps
        axes[0].plot(df['steps'], df['ppl_improvement'], marker='o', linewidth=3, markersize=10, color='blue')
        axes[0].set_title('PPL Improvement vs Inference Steps', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Inference Steps', fontsize=12)
        axes[0].set_ylabel('PPL Improvement', fontsize=12)
        axes[0].set_xticks(df['steps'])
        axes[0].set_xticklabels([f'{int(x)}' for x in df['steps']])
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Self-accuracy vs Inference Steps
        if df['self_accuracy'].notna().any():
            axes[1].plot(df['steps'], df['self_accuracy'], marker='s', linewidth=3, markersize=10, color='orange')
            axes[1].set_title('Self-Accuracy vs Inference Steps', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Inference Steps', fontsize=12)
            axes[1].set_ylabel('Self-Accuracy', fontsize=12)
            axes[1].set_xticks(df['steps'])
            axes[1].set_xticklabels([f'{int(x)}' for x in df['steps']])
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: PPL improvement ratio vs Inference Steps
        if df['ppl_ratio'].notna().any():
            axes[2].plot(df['steps'], df['ppl_ratio'], marker='^', linewidth=3, markersize=10, color='green')
            axes[2].set_title('PPL Improvement Ratio vs Inference Steps', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Inference Steps', fontsize=12)
            axes[2].set_ylabel('Improvement Ratio', fontsize=12)
            axes[2].set_xticks(df['steps'])
            axes[2].set_xticklabels([f'{int(x)}' for x in df['steps']])
            axes[2].grid(True, alpha=0.3)
        
        # No surprisal plot
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Improvement metrics plot saved to {save_path}")
        
    def plot_entropy_analysis(self, save_path: str = "plots/inference_steps_entropy.png"):
        """Plot entropy analysis across different inference steps"""
        # Prepare data
        steps_data = []
        for steps, info in self.metrics_data.items():
            data = info['data']
            
            # Extract entropy metrics
            entropy_metrics = data.get('entropy_metrics', {})
            entropy_improvements = data.get('entropy_improvements', {})
            
            if entropy_metrics:
                steps_data.append({
                    'steps': steps,
                    'ent_per_seq': entropy_metrics.get('ent_per_seq', None),
                    'ent_per_token': entropy_metrics.get('ent_per_token', None),
                    'seq_change': entropy_improvements.get('seq_change', None),
                    'token_change': entropy_improvements.get('token_change', None)
                })
        
        if not steps_data:
            print("No entropy metrics data found for plotting")
            return
            
        df = pd.DataFrame(steps_data)
        df = df.sort_values('steps')
        
        # Create separate plots for each metric
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # Plot 1: Entropy per sequence vs Inference Steps
        if df['ent_per_seq'].notna().any():
            axes[0].plot(df['steps'], df['ent_per_seq'], marker='o', linewidth=3, markersize=10, color='blue')
            axes[0].set_title('Entropy per Sequence vs Inference Steps', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Inference Steps', fontsize=12)
            axes[0].set_ylabel('Entropy per Sequence', fontsize=12)
            axes[0].set_xticks(df['steps'])
            axes[0].set_xticklabels([f'{int(x)}' for x in df['steps']])
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Entropy per token vs Inference Steps
        if df['ent_per_token'].notna().any():
            axes[1].plot(df['steps'], df['ent_per_token'], marker='s', linewidth=3, markersize=10, color='orange')
            axes[1].set_title('Entropy per Token vs Inference Steps', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Inference Steps', fontsize=12)
            axes[1].set_ylabel('Entropy per Token', fontsize=12)
            axes[1].set_xticks(df['steps'])
            axes[1].set_xticklabels([f'{int(x)}' for x in df['steps']])
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Sequence entropy change vs Inference Steps
        if df['seq_change'].notna().any():
            axes[2].plot(df['steps'], df['seq_change'], marker='^', linewidth=3, markersize=10, color='green')
            axes[2].set_title('Sequence Entropy Change vs Inference Steps', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Inference Steps', fontsize=12)
            axes[2].set_ylabel('Entropy Change', fontsize=12)
            axes[2].set_xticks(df['steps'])
            axes[2].set_xticklabels([f'{int(x)}' for x in df['steps']])
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Token entropy change vs Inference Steps
        if df['token_change'].notna().any():
            axes[3].plot(df['steps'], df['token_change'], marker='d', linewidth=3, markersize=10, color='red')
            axes[3].set_title('Token Entropy Change vs Inference Steps', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Inference Steps', fontsize=12)
            axes[3].set_ylabel('Entropy Change', fontsize=12)
            axes[3].set_xticks(df['steps'])
            axes[3].set_xticklabels([f'{int(x)}' for x in df['steps']])
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Entropy analysis plot saved to {save_path}")
        
    def plot_bleu_scores(self, save_path: str = "plots/inference_steps_bleu.png"):
        """Plot BLEU scores across different inference steps"""
        # Prepare data
        steps_data = []
        for steps, info in self.metrics_data.items():
            data = info['data']
            
            # Extract BLEU metrics
            bleu_metrics = data.get('bleu', {})
            corrected_vs_original = bleu_metrics.get('corrected_vs_original', {})
            
            if corrected_vs_original:
                steps_data.append({
                    'steps': steps,
                    'bleu_score': corrected_vs_original.get('bleu', None)
                })
        
        if not steps_data:
            print("No BLEU metrics data found for plotting")
            return
            
        df = pd.DataFrame(steps_data)
        df = df.sort_values('steps')
        
        # Create separate plots for each metric
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: BLEU score vs Inference Steps
        if df['bleu_score'].notna().any():
            axes[0].plot(df['steps'], df['bleu_score'], marker='o', linewidth=3, markersize=10, color='blue')
            axes[0].set_title('BLEU Score vs Inference Steps', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Inference Steps', fontsize=12)
            axes[0].set_ylabel('BLEU Score', fontsize=12)
            axes[0].set_xticks(df['steps'])
            axes[0].set_xticklabels([f'{int(x)}' for x in df['steps']])
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: BLEU score improvement (if we have baseline)
        if len(df) > 1 and df['bleu_score'].notna().any():
            baseline_bleu = df.iloc[0]['bleu_score']  # Use first step as baseline
            df['bleu_improvement'] = df['bleu_score'] - baseline_bleu
            axes[1].plot(df['steps'], df['bleu_improvement'], marker='s', linewidth=3, markersize=10, color='orange')
            axes[1].set_title('BLEU Score Improvement vs Inference Steps', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Inference Steps', fontsize=12)
            axes[1].set_ylabel('BLEU Improvement', fontsize=12)
            axes[1].set_xticks(df['steps'])
            axes[1].set_xticklabels([f'{int(x)}' for x in df['steps']])
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"BLEU scores plot saved to {save_path}")
        
    def generate_summary_table(self, save_path: str = "plots/inference_steps_summary.csv"):
        """Generate a summary table of all metrics for different inference steps"""
        summary_data = []
        
        for steps, info in self.metrics_data.items():
            data = info['data']
            
            # Extract all relevant metrics
            external_metrics = data.get('external_metrics', {})
            self_ppl_metrics = data.get('self_ppl_metrics', {})
            ppl_improvement = data.get('self_ppl_improvement', {})
            entropy_metrics = data.get('entropy_metrics', {})
            entropy_improvements = data.get('entropy_improvements', {})
            bleu_metrics = data.get('bleu', {})
            
            summary_data.append({
                'inference_steps': steps,
                'external_ppl': external_metrics.get('ppl', None),
                'external_accuracy': external_metrics.get('accuracy', None),
                'self_ppl': self_ppl_metrics.get('average_perplexity', None),
                'ppl_improvement': ppl_improvement.get('absolute_improvement', None),
                'ppl_improvement_ratio': ppl_improvement.get('improvement_ratio', None),
                'self_accuracy': data.get('average_self_accuracy', None),
                'entropy_per_seq': entropy_metrics.get('ent_per_seq', None),
                'entropy_per_token': entropy_metrics.get('ent_per_token', None),
                'seq_entropy_change': entropy_improvements.get('seq_change', None),
                'token_entropy_change': entropy_improvements.get('token_change', None),
                'bleu_score': bleu_metrics.get('corrected_vs_original', {}).get('bleu', None)
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.sort_values('inference_steps')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Summary table saved to {save_path}")
            print("\nSummary Table:")
            print(df.to_string(index=False))
        else:
            print("No data found for summary table")
            
    def plot_all(self):
        """Generate all plots"""
        print("Generating all inference steps comparison plots...")
        self.plot_perplexity_comparison()
        self.plot_improvement_metrics()
        self.plot_entropy_analysis()
        self.plot_bleu_scores()
        self.generate_summary_table()
        print("All inference steps comparison plots generated successfully!")

def main():
    parser = argparse.ArgumentParser(description='Plot metrics comparison across different inference steps')
    parser.add_argument('--samples_dir', default='Samples', help='Directory containing metrics files')
    parser.add_argument('--plot_type', choices=['all', 'perplexity', 'improvement', 'entropy', 'bleu', 'summary'], 
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    analyzer = InferenceStepsAnalyzer(args.samples_dir)
    analyzer.load_inference_steps_metrics()
    
    if args.plot_type == 'all':
        analyzer.plot_all()
    elif args.plot_type == 'perplexity':
        analyzer.plot_perplexity_comparison()
    elif args.plot_type == 'improvement':
        analyzer.plot_improvement_metrics()
    elif args.plot_type == 'entropy':
        analyzer.plot_entropy_analysis()
    elif args.plot_type == 'bleu':
        analyzer.plot_bleu_scores()
    elif args.plot_type == 'summary':
        analyzer.generate_summary_table()

if __name__ == "__main__":
    main()
