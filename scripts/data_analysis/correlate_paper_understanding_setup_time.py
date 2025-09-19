# scripts/data_analysis/correlate_paper_understanding_setup_time.py

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr, linregress
import os
import sys

sys.path.append('.')
from constants import SUBJECT_DATA, PAPERS, PAPER_TITLES, SKILL_LEVELS, SKILL_LEVEL_TITLES, \
                      SKILL_LEVEL_COLORS, PAPER_COLORS, GRAPH_OUTPUT_DIR
from graph import GraphData, scatter_plot_2d

# =========================
# 1) Coleta dos dados
# =========================
x_points = []          # compreensão (%)  -- no seu JSON está em percentuais (ex.: 72, 90)
y_points = []          # tempo de setup (min)
y_points2 = []         # nota de dificuldade do setup (1-5)
point_categories = []  # nível (novice/intermediate/advanced)
point_categories2 = [] # paper (A/B/C)

for group in PAPERS:
    for student in SUBJECT_DATA:
        if student.paper == group:
            setup_time_training = student.postsurvey["experiment_time_setup_training_hours"] * 60 \
                                + student.postsurvey["experiment_time_setup_training_minutes"]
            setup_time_evaluation = student.postsurvey["experiment_time_setup_evaluation_hours"] * 60 \
                                  + student.postsurvey["experiment_time_setup_evaluation_minutes"]
            setup_mins = setup_time_training + setup_time_evaluation

            paper_understanding_accuracy = student.postsurvey['understanding_cq_accuracy']  # já em %
            code_setup_rating = student.postsurvey['experiment_rating_setup']               # 1-5 (invertido em constants.py)

            x_points.append(paper_understanding_accuracy)
            y_points.append(setup_mins)
            y_points2.append(code_setup_rating)
            point_categories.append(student.skill_level)
            point_categories2.append(student.paper)

# =========================
# 2) Correlações (impressas no terminal)
# =========================
corr, p = pearsonr(x_points, y_points)
print('Comprehension Q accuracy vs. setup time (%s) Pearson correlation: %.3f (p=%.5f)' % ('all', corr, p))

for group in SKILL_LEVELS:
    x_group = [x for x, g in zip(x_points, point_categories) if g == group]
    y_group = [y for y, g in zip(y_points, point_categories) if g == group]

    # Pode dar NaN/1.0 em amostra muito pequena; é esperado no dataset sintético
    corr_aux_spearman, p_aux_spearman = spearmanr(x_group, y_group)
    print('Comprehension Q accuracy vs. setup time (%s) Spearman correlation: %.3f (p=%.5f)' % (group, corr_aux_spearman, p_aux_spearman))

    corr_aux_pearson, p_aux_pearson = pearsonr(x_group, y_group)
    print('Comprehension Q accuracy vs. setup time (%s) Pearson correlation: %.3f (p=%.5f)' % (group, corr_aux_pearson, p_aux_pearson))

    slope, intercept, corr_aux_linreg, p_aux_linreg, slope_err = linregress(x_group, y_group)
    print('Comprehension Q accuracy vs. setup time (%s) linear regression: slope=%.5f (+- %.5f), intercept=%.5f, r=%.5f (p=%.5f)'
          % (group, slope, slope_err, intercept, corr_aux_linreg, p_aux_linreg))

# =========================
# 3) Faixas e ticks (X em %; Y em min)
# =========================
# X (%): definimos uma janela um pouco maior que os dados (padding de 5%)
x_min, x_max = min(x_points), max(x_points)
x_lo = max(0, (int(x_min) // 5) * 5 - 5)           # arredonda para baixo e abre -5
x_hi = min(100, ((int(x_max) + 4) // 5) * 5 + 5)   # arredonda para cima e abre +5
x_ticks = list(np.arange(x_lo, x_hi + 1, 5 if (x_hi - x_lo) <= 50 else 10))

# Y (min): 0–200 cobre seus dados (≈70–150)
y_range = (0, 200)
y_ticks = list(np.arange(0, 201, 25))

# =========================
# 4) Gráfico 1 — por NÍVEL
# =========================
graph_data = GraphData(
    x=x_points,
    y=y_points,
    x_label='Paper Comprehension (%)',
    y_label='Setup Time (min.)',
    x_range=(x_lo, x_hi),
    y_range=y_range,
    x_ticks=x_ticks,
    y_ticks=y_ticks,
    colors=[SKILL_LEVEL_COLORS[group] for group in point_categories],
    point_categories=[SKILL_LEVEL_TITLES[group] for group in point_categories],
)
scatter_plot_2d(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "understanding_vs_setup_time_by_skill_level.pdf"))

# =========================
# 5) Gráfico 2 — por PAPER
# =========================
graph_data = GraphData(
    x=x_points,
    y=y_points,
    x_label='Paper Comprehension (%)',
    y_label='Setup Time (min.)',
    x_range=(x_lo, x_hi),
    y_range=y_range,
    x_ticks=x_ticks,
    y_ticks=y_ticks,
    colors=[PAPER_COLORS[group] for group in point_categories2],
    point_categories=[PAPER_TITLES[group] for group in point_categories2],
)
scatter_plot_2d(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "understanding_vs_setup_time_by_paper.pdf"))

# =========================
# 6) Gráfico 3 — compreensão × dificuldade percebida (por NÍVEL)
# =========================
graph_data = GraphData(
    x=x_points,
    y=y_points2,
    x_label='Paper Comprehension (%)',
    y_label='Setup Difficulty (1–5)',
    x_range=(x_lo, x_hi),
    y_range=(0, 5.5),
    x_ticks=x_ticks,
    y_ticks=list(np.arange(0, 5.5, 1.0)),
    colors=[SKILL_LEVEL_COLORS[group] for group in point_categories],
    point_categories=[SKILL_LEVEL_TITLES[group] for group in point_categories],
)
scatter_plot_2d(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "understanding_vs_setup_rating_by_skill_level.pdf"))

