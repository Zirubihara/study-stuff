# Standardization Implementation Plan

## Current State Analysis

### Matplotlib (12 charts - GOOD but needs renaming)
**Data Processing:**
- execution_time_comparison.png → RENAME to dp_execution_time.png
- operation_breakdown_10M.png → RENAME to dp_operation_breakdown.png
- memory_usage_comparison.png → RENAME to dp_memory_usage.png
- scalability_analysis.png → RENAME to dp_scalability.png
- performance_rankings_10M.png → RENAME to dp_performance_rankings.png
- summary_table_10M.png → RENAME to dp_summary_table.png

**ML Frameworks:**
- ml_training_time_comparison.png → RENAME to ml_training_time.png
- ml_inference_speed_comparison.png → RENAME to ml_inference_speed.png
- ml_memory_usage_comparison.png → RENAME to ml_memory_usage.png
- ml_anomaly_rate_comparison.png → RENAME to ml_anomaly_rate.png
- ml_framework_comparison_matrix.png → RENAME to ml_comparison_matrix.png
- ml_performance_summary_table.png → RENAME to ml_summary_table.png

### Plotly (10 charts - needs 2 more + renaming)
**Current:**
- execution_time_interactive.html
- operation_breakdown_stacked_10M.html
- operation_heatmap.html
- scalability_interactive.html
- performance_radar_10M.html
- memory_vs_time_scatter.html
- ml_training_vs_inference_interactive.html
- ml_framework_radar_interactive.html
- ml_multi_metric_comparison.html
- ml_framework_ranking_interactive.html

**MISSING:**
- dp_memory_usage.html
- dp_summary_table.html

**Need to rename all to match standard**

### Bokeh (6 charts - needs 6 more)
**Current:**
- dp_execution_time_10M.html
- dp_operation_breakdown_10M.html
- dp_scalability_analysis.html
- ml_training_time.html
- ml_inference_speed.html
- ml_memory_usage.html

**MISSING:**
- dp_memory_usage.html
- dp_performance_rankings.html
- dp_summary_table.html
- ml_anomaly_rate.html
- ml_comparison_matrix.html
- ml_summary_table.html

### Holoviews (6 charts - needs 6 more)
**Current:**
- dp_execution_time_10M.html
- dp_operation_breakdown_10M.html
- dp_scalability_analysis.html
- ml_training_time.html
- ml_inference_speed.html
- ml_comparison_heatmap.html

**MISSING:**
- dp_memory_usage.html
- dp_performance_rankings.html
- dp_summary_table.html
- ml_memory_usage.html
- ml_anomaly_rate.html
- ml_summary_table.html

## Implementation Strategy

### Phase 1: Quick Fix (Recommended)
Update output file names in existing scripts without changing logic.

#### Matplotlib - Quick File Name Updates
In `data_processing_visualization.py`:
- Line ~81: Change to `'dp_execution_time.png'`
- Line ~117: Change to `'dp_operation_breakdown.png'`
- Line ~164: Change to `'dp_memory_usage.png'`
- Line ~204: Change to `'dp_scalability.png'`
- Line ~252: Change to `'dp_performance_rankings.png'`
- Line ~306: Change to `'dp_summary_table.png'`

In `ml_frameworks_visualization.py`:
- Line ~94: Change to `'ml_training_time.png'`
- Line ~147: Change to `'ml_inference_speed.png'`
- Line ~199: Change to `'ml_memory_usage.png'`
- Line ~253: Change to `'ml_anomaly_rate.png'`
- Line ~318: Change to `'ml_comparison_matrix.png'`
- Line ~377: Change to `'ml_summary_table.png'`

#### Plotly - Add Missing + Rename
Need to add two methods in `data_processing_visualization.py`:
1. `plot_memory_usage()` - simple bar chart
2. `plot_summary_table()` - table visualization

Then rename all output files to match standard.

#### Bokeh - Add 6 Missing Charts
Need to add 6 new methods to `combined_visualization.py`:
1. `plot_dp_memory_usage()`
2. `plot_dp_performance_rankings()`
3. `plot_dp_summary_table()`
4. `plot_ml_anomaly_rate()`
5. `plot_ml_comparison_matrix()`
6. `plot_ml_summary_table()`

#### Holoviews - Add 6 Missing Charts
Same as Bokeh - add 6 missing methods.

### Phase 2: Verification
After updates:
1. Run each library's visualization scripts
2. Check output/ directories for exactly 12 files each
3. Verify file names match specification
4. Visual inspection for content consistency

### Phase 3: Master Script Update
Update `generate_all_visualizations.py` to:
1. Run both data_processing and ml_frameworks scripts for each library
2. Verify 12 files created per library
3. Report any missing files

## Estimated Time
- Phase 1: 30-45 minutes (mostly find-replace)
- Phase 2: 15 minutes (testing)
- Phase 3: 10 minutes (script update)

**Total: ~1 hour**

## Quick Start Commands

After standardization, users should be able to:

```bash
# Generate all visualizations
cd data_visualization
python generate_all_visualizations.py

# Result: 48 files created
#   matplotlib/output/  (12 PNG files)
#   plotly/output/      (12 HTML files)
#   bokeh/output/       (12 HTML files)
#   holoviews/output/   (12 HTML files)
```

## File Name Reference Card

### Data Processing (6 files)
| Standard Name | Description |
|---------------|-------------|
| dp_execution_time | Total execution time bar chart |
| dp_operation_breakdown | Grouped bar chart by operation |
| dp_memory_usage | Memory usage bar chart |
| dp_scalability | Line chart (log-log scale) |
| dp_performance_rankings | 6-panel operation rankings |
| dp_summary_table | Comprehensive data table |

### ML/DL Frameworks (6 files)
| Standard Name | Description |
|---------------|-------------|
| ml_training_time | Training time bar chart |
| ml_inference_speed | Inference speed bar chart |
| ml_memory_usage | Memory usage bar chart |
| ml_anomaly_rate | Anomaly detection rate chart |
| ml_comparison_matrix | 2x2 grid of all metrics |
| ml_summary_table | Comprehensive data table |

## Next Action
Would you like me to:
1. Create the updated scripts automatically (I'll make the changes)
2. Provide step-by-step manual instructions
3. Create a simple find-replace script

Choose option 1 for fastest results.
