"""仅运行图9: 2季节x3场景 优化前后传播曲线对比图"""
import sys
sys.path.insert(0, ".")

from paper_figures import plot_season_scenario_grid, FIG_DIR

print("仅生成图9: 2季节x3场景 对比图")
print("=" * 50)
plot_season_scenario_grid()
print("\n完成! 输出:", FIG_DIR / "fig9_season_scenario_grid.png")
