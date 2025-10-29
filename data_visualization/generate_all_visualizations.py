"""
Master Script to Generate All Visualizations - ALL 5 FRAMEWORKS

Run this to create 95+ charts across all 5 visualization frameworks:
- Matplotlib (24 PNG files @ 300 DPI) - For thesis document
- Plotly (22 HTML files) - Interactive web charts  
- Bokeh (24 HTML files) - Interactive dashboards
- Holoviews (25 HTML files) - Declarative visualizations
- Streamlit (Live dashboard) - Run separately for live demos

Total: 95+ visualizations ready for thesis use!
"""

import subprocess
import sys
from pathlib import Path
import time
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def run_visualization(lib_name, script_path, description):
    """Run a single visualization script"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ SUCCESS ({elapsed:.1f}s): {lib_name} visualizations generated")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        print(f"   {line}")
            return True
        else:
            print(f"❌ FAILED ({elapsed:.1f}s): {lib_name}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: Exceeded 5 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Generate all visualizations"""
    print("="*80)
    print("GENERATING ALL VISUALIZATIONS FOR THESIS - ALL 5 FRAMEWORKS")
    print("="*80)
    print("\nThis will generate:")
    print("  - 24 Matplotlib PNG charts (publication quality)")
    print("  - 22 Plotly HTML charts (interactive)")
    print("  - 6 Plotly Operation-Specific charts (interactive)")
    print("  - 24 Bokeh HTML charts (interactive)")
    print("  - 25 Holoviews HTML charts (interactive)")
    print("  - Streamlit dashboard (live web application)")
    print("\nTotal: 95+ visualizations across all 5 frameworks")
    print("="*80)

    scripts = [
        ("Matplotlib", "matplotlib/data_processing_visualization.py",
         "Matplotlib - Data Processing Comparison"),
        ("Matplotlib", "matplotlib/ml_frameworks_visualization.py",
         "Matplotlib - ML/DL Frameworks Comparison"),
        ("Plotly", "plotly/data_processing_visualization.py",
         "Plotly - Data Processing Comparison"),
        ("Plotly", "plotly/ml_frameworks_visualization.py",
         "Plotly - ML/DL Frameworks Comparison"),
        ("Plotly", "plotly/operation_specific_charts.py",
         "Plotly - Operation-Specific Charts"),
        ("Bokeh", "bokeh/combined_visualization.py",
         "Bokeh - Combined Visualizations"),
        ("Holoviews", "holoviews/combined_visualization.py",
         "Holoviews - Combined Visualizations"),
    ]

    results = {}
    total_start = time.time()

    for lib, script, desc in scripts:
        if Path(script).exists():
            results[desc] = run_visualization(lib, script, desc)
        else:
            print(f"\n⚠️  SKIP: {script} not found")
            results[desc] = None

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)

    successes = 0
    failures = 0
    skipped = 0

    for desc, status in results.items():
        if status is True:
            print(f"✅ {desc}")
            successes += 1
        elif status is False:
            print(f"❌ {desc}")
            failures += 1
        else:
            print(f"⚠️  {desc} - Not found")
            skipped += 1

    print("\n" + "="*80)
    print(f"Results: {successes} succeeded, {failures} failed, {skipped} skipped")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print("="*80)

    # Output locations
    if successes > 0:
        print("\n📊 Generated visualizations are in:")
        print("  - matplotlib/output/  (24 PNG files @ 300 DPI for thesis document)")
        print("  - plotly/output/      (22 HTML files for interactive viewing)")
        print("  - bokeh/output/       (24 HTML files for interactive viewing)")
        print("  - holoviews/output/   (25 HTML files for interactive viewing)")
        print("\n  Total: 95+ static/interactive visualizations generated!")

    # Streamlit info
    print("\n" + "="*80)
    print("FRAMEWORK 5/5: STREAMLIT DASHBOARD (Live Web Application)")
    print("="*80)
    print("Streamlit is a live web application framework (not a file generator).")
    print("\n🚀 To run the interactive dashboard:")
    print("\n  Option 1 - Full Dashboard:")
    print("    cd streamlit")
    print("    streamlit run dashboard.py")
    print("\n  Option 2 - From root:")
    print("    streamlit run streamlit/dashboard.py")
    print("\n📝 For thesis comparison (side-by-side with other frameworks):")
    print("    python comparative_visualization_thesis.py")
    print("    # Generates Streamlit code in THESIS_COMPARISON_CHARTS/streamlit/")
    print("\n💡 Use Streamlit for:")
    print("  - Live thesis defense demonstrations")
    print("  - Interactive committee Q&A sessions")
    print("  - Real-time data filtering and exploration")
    print("="*80)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
