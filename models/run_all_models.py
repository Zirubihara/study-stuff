"""
Unified Runner for All Anomaly Detection Frameworks

This script runs all 5 ML/DL frameworks sequentially and generates comparison visualizations.

Frameworks:
1. Scikit-learn (Isolation Forest + LOF)
2. XGBoost (Gradient Boosting)
3. PyTorch (MLP Autoencoder)
4. TensorFlow (MLP Autoencoder)
5. JAX (MLP Autoencoder with JIT)

Usage:
    python run_all_models.py
    python run_all_models.py --skip sklearn,pytorch  # Skip specific frameworks
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def run_framework(name, script_path, description):
    """
    Run a single framework script

    Args:
        name: Framework name
        script_path: Path to the Python script
        description: Framework description

    Returns:
        tuple: (success: bool, execution_time: float)
    """
    print(f"\n{Colors.BLUE}{Colors.BOLD}▶ Running {name}...{Colors.ENDC}")
    print(f"  {description}")
    print(f"  Script: {script_path}")

    if not Path(script_path).exists():
        print_error(f"Script not found: {script_path}")
        return False, 0.0

    start_time = time.time()

    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=Path(script_path).parent,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print_success(f"{name} completed successfully in {execution_time:.2f}s")
            return True, execution_time
        else:
            print_error(f"{name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"\n{Colors.RED}Error output:{Colors.ENDC}")
                print(result.stderr[:500])  # Print first 500 chars of error
            return False, execution_time

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print_error(f"{name} timed out after {execution_time:.2f}s")
        return False, execution_time

    except Exception as e:
        execution_time = time.time() - start_time
        print_error(f"{name} failed with exception: {str(e)}")
        return False, execution_time


def run_visualization():
    """Run the comparison visualization script"""
    print(
        f"\n{Colors.BLUE}{Colors.BOLD}▶ Generating comparison visualizations...{Colors.ENDC}"
    )

    viz_script = "visualization/compare_all_results.py"

    if not Path(viz_script).exists():
        print_error(f"Visualization script not found: {viz_script}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, viz_script],
            cwd=Path(viz_script).parent,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        if result.returncode == 0:
            print_success("Comparison visualizations generated successfully")
            print_info("Charts saved to: models/charts/")
            return True
        else:
            print_error("Visualization generation failed")
            if result.stderr:
                print(result.stderr[:500])
            return False

    except Exception as e:
        print_error(f"Visualization failed with exception: {str(e)}")
        return False


def load_all_results():
    """Load all framework results for summary"""
    results = {}
    results_dir = Path("results")

    result_files = {
        "sklearn": "sklearn_anomaly_detection_results.json",
        "pytorch": "pytorch_anomaly_detection_results.json",
        "tensorflow": "tensorflow_anomaly_detection_results.json",
        "jax": "jax_anomaly_detection_results.json",
        "xgboost": "xgboost_anomaly_detection_results.json",
    }

    for framework, filename in result_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    results[framework] = json.load(f)
            except Exception as e:
                print_warning(f"Could not load {framework} results: {e}")

    return results


def print_summary(execution_results, results_data):
    """Print a summary of all results"""
    print_header("EXECUTION SUMMARY")

    # Execution status
    print(f"{Colors.BOLD}Framework Execution Status:{Colors.ENDC}")
    total_time = 0.0
    successful = 0

    for name, (success, exec_time) in execution_results.items():
        status = (
            f"{Colors.GREEN}✓ SUCCESS{Colors.ENDC}"
            if success
            else f"{Colors.RED}✗ FAILED{Colors.ENDC}"
        )
        print(f"  {name:20s} {status}  ({exec_time:.2f}s)")
        total_time += exec_time
        if success:
            successful += 1

    print(
        f"\n{Colors.BOLD}Total Execution Time: {total_time:.2f}s ({total_time/60:.2f} minutes){Colors.ENDC}"
    )
    print(
        f"{Colors.BOLD}Success Rate: {successful}/{len(execution_results)} frameworks{Colors.ENDC}"
    )

    # Performance summary from results
    if results_data:
        print(f"\n{Colors.BOLD}Performance Metrics Summary:{Colors.ENDC}\n")

        # Table header
        print(
            f"{'Framework':<20} {'Train Time':<12} {'Inference':<12} {'Speed':<15} {'Anomalies':<12}"
        )
        print("-" * 80)

        # Sklearn results (two models)
        if "sklearn" in results_data:
            sklearn = results_data["sklearn"]

            if "isolation_forest" in sklearn:
                if_data = sklearn["isolation_forest"]
                print(
                    f"{'Isolation Forest':<20} "
                    f"{if_data['training_time']:>10.2f}s  "
                    f"{if_data['inference_time']:>10.2f}s  "
                    f"{if_data['inference_speed']:>12,.0f}/s  "
                    f"{if_data['n_anomalies']:>8,}"
                )

            if "local_outlier_factor" in sklearn:
                lof_data = sklearn["local_outlier_factor"]
                print(
                    f"{'LOF':<20} "
                    f"{lof_data['training_time']:>10.2f}s  "
                    f"{lof_data['inference_time']:>10.2f}s  "
                    f"{lof_data['inference_speed']:>12,.0f}/s  "
                    f"{lof_data['n_anomalies']:>8,}"
                )

        # Other frameworks
        for framework, key in [
            ("pytorch", "pytorch_autoencoder"),
            ("tensorflow", "tensorflow_autoencoder"),
            ("jax", "jax_autoencoder"),
            ("xgboost", "xgboost_detector"),
        ]:
            if framework in results_data and key in results_data[framework]:
                data = results_data[framework][key]
                display_name = framework.capitalize()
                print(
                    f"{display_name:<20} "
                    f"{data['training_time']:>10.2f}s  "
                    f"{data['inference_time']:>10.2f}s  "
                    f"{data['inference_speed']:>12,.0f}/s  "
                    f"{data['n_anomalies']:>8,}"
                )

        print()

    # Next steps
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print(f"  1. View comparison charts: {Colors.CYAN}models/charts/{Colors.ENDC}")
    print(f"  2. Check detailed results: {Colors.CYAN}models/results/{Colors.ENDC}")
    print(
        f"  3. Review framework logs: {Colors.CYAN}models/[framework]/*.log{Colors.ENDC}"
    )


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run all anomaly detection frameworks")
    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-separated list of frameworks to skip (e.g., sklearn,pytorch)",
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Comma-separated list of frameworks to run (e.g., sklearn,xgboost)",
    )
    args = parser.parse_args()

    # Define all frameworks
    frameworks = {
        "sklearn": {
            "script": "sklearn/anomaly_detection_sklearn.py",
            "description": "Classical ML: Isolation Forest + Local Outlier Factor",
        },
        "xgboost": {
            "script": "xgboost/anomaly_detection_xgboost.py",
            "description": "Gradient Boosting: XGBoost Anomaly Detector",
        },
        "pytorch": {
            "script": "pytorch/anomaly_detection_pytorch.py",
            "description": "Deep Learning: PyTorch MLP Autoencoder",
        },
        "tensorflow": {
            "script": "tensorflow/anomaly_detection_tensorflow.py",
            "description": "Deep Learning: TensorFlow MLP Autoencoder",
        },
        "jax": {
            "script": "jax/anomaly_detection_jax.py",
            "description": "Deep Learning: JAX MLP Autoencoder (JIT-compiled)",
        },
    }

    # Filter frameworks based on arguments
    skip_list = set(args.skip.split(",")) if args.skip else set()
    only_list = set(args.only.split(",")) if args.only else None

    if only_list:
        frameworks_to_run = {k: v for k, v in frameworks.items() if k in only_list}
    else:
        frameworks_to_run = {k: v for k, v in frameworks.items() if k not in skip_list}

    # Print startup banner
    print_header("ANOMALY DETECTION FRAMEWORK BENCHMARK")
    print(f"{Colors.BOLD}Running {len(frameworks_to_run)} frameworks:{Colors.ENDC}")
    for name in frameworks_to_run.keys():
        print(f"  • {name.capitalize()}")

    if skip_list:
        print(f"\n{Colors.YELLOW}Skipping: {', '.join(skip_list)}{Colors.ENDC}")

    print(f"\n{Colors.CYAN}Data: Japanese Trade Dataset (1988-2020){Colors.ENDC}")
    print(f"{Colors.CYAN}Sample Size: 5M rows{Colors.ENDC}")
    print(
        f"{Colors.CYAN}Expected Duration: 5-15 minutes (depending on hardware){Colors.ENDC}"
    )

    input(f"\n{Colors.BOLD}Press Enter to start...{Colors.ENDC}")

    # Run all frameworks
    overall_start = time.time()
    execution_results = {}

    for name, config in frameworks_to_run.items():
        success, exec_time = run_framework(
            name, config["script"], config["description"]
        )
        execution_results[name] = (success, exec_time)

        # Small delay between frameworks
        time.sleep(1)

    # Generate visualizations if at least one framework succeeded
    successful_count = sum(1 for success, _ in execution_results.values() if success)

    if successful_count > 0:
        viz_success = run_visualization()
    else:
        print_error("No frameworks completed successfully. Skipping visualization.")
        viz_success = False

    overall_time = time.time() - overall_start

    # Load results and print summary
    results_data = load_all_results()
    print_summary(execution_results, results_data)

    # Final status
    print_header("BENCHMARK COMPLETE")
    print(
        f"{Colors.BOLD}Total Time: {overall_time:.2f}s ({overall_time/60:.2f} minutes){Colors.ENDC}"
    )

    if successful_count == len(frameworks_to_run):
        print_success("All frameworks completed successfully!")
        if viz_success:
            print_success("Comparison visualizations generated!")
        return 0
    elif successful_count > 0:
        print_warning(
            f"{successful_count}/{len(frameworks_to_run)} frameworks completed successfully"
        )
        return 1
    else:
        print_error("All frameworks failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Benchmark interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
