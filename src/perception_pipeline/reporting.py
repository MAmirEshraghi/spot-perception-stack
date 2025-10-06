import os
import json
import numpy as np
import matplotlib.pyplot as plt

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def analyze_and_save_timing_log(perception_log, timing_log, process_limit):
    """Saves raw timing data, generates plots, and prints/saves a detailed summary."""
    print("\n--- 5. Analyzing Performance and Generating Reports ---")
    
    timing_dir = os.path.join(perception_log.scan_dir, "timing_analysis")
    os.makedirs(timing_dir, exist_ok=True)

    json_path = os.path.join(timing_dir, "timing_log.json")
    with open(json_path, "w") as f:
        json.dump(timing_log, f, indent=4, cls=NumpyEncoder)
    print(f"  > Raw timing data saved to: {json_path}")

    # Plot generation
    for key, value in timing_log.items():
        if isinstance(value, list) and len(value) > 1:
            try:
                y_data, y_label, unit = [], "", ""
                if "time_ms" in value[0]:
                    y_data, y_label, unit = [d['time_ms'] for d in value], "Time (ms)", "ms"
                elif "mask_count" in value[0]:
                    y_data, y_label, unit = [d['mask_count'] for d in value], "Number of Masks", ""
                else: continue

                plt.figure(figsize=(15, 7))
                plt.plot(range(len(y_data)), y_data, marker='o', linestyle='-', markersize=4)
                mean_val = np.mean(y_data)
                plt.axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}{unit}')
                plt.title(f"Performance for '{key}'"); plt.xlabel("Sample Index"); plt.ylabel(y_label)
                plt.grid(True); plt.legend(); plt.tight_layout()
                plot_path = os.path.join(timing_dir, f"{key.replace(' ', '_')}_performance.png")
                plt.savefig(plot_path); plt.close()
                print(f"  > Performance plot saved to: {plot_path}")
            except Exception as e:
                print(f"  > Could not generate plot for '{key}': {e}")
    
    # summary report generation
    summary_lines = ["="*80, " " * 28 + "PERFORMANCE SUMMARY", "="*80]
    limit_str = f"Image Process Limit: {process_limit if process_limit else 'None'}"
    summary_lines.append(f"{'Run Configuration':<35}: {limit_str}")
    summary_lines.append("-" * 80)
    for key, value in timing_log.items():
        if isinstance(value, list) and value:
            data_list, unit = [], ""
            if "time_ms" in value[0]: data_list, unit = [d['time_ms'] for d in value], "ms"
            elif "mask_count" in value[0]: data_list, unit = [d['mask_count'] for d in value], ""
            stats = { "avg": np.mean(data_list), "std": np.std(data_list), "min": np.min(data_list), "max": np.max(data_list) }
            line = (f"{key:<35}: avg={stats['avg']:.3f}{unit}, std={stats['std']:.3f}{unit}, min={stats['min']:.3f}{unit}, max={stats['max']:.3f}{unit} ({len(data_list)} samples)")
            summary_lines.append(line)
        elif not isinstance(value, list):
            summary_lines.append(f"{key:<35}: {value*1000:.3f} ms")
    summary_lines.append("="*80)
    
    summary_text = "\n".join(summary_lines)
    print("\n\n" + summary_text)
    summary_path = os.path.join(perception_log.scan_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\n> Performance summary saved to: {summary_path}")