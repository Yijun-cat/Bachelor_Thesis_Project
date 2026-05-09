import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
OUTPUT_STYLES = {
    "tracking_error": {"color": "tab:blue", "marker": "o", "label": "Tracking error"},
    "rmse": {"color": "tab:yellow", "marker": "s", "label": "RMSE"},
    "reaction_time": {"color": "tab:green", "marker": "^", "label": "Reaction time"},
    "total_error": {"color": "tab:red", "marker": "D", "label": "Total error"},
}


def save_name(base_name, lag_feature=False, ext="png"):
    if lag_feature:
        return f"{base_name}_lag.{ext}"
    return f"{base_name}.{ext}"


def plot_rmse_bar(
    df_metrics,
    x_col,
    y_col,
    x_label,
    y_label,
    title,
    out_dir,
    file_stub,
    lag_feature=False,
    rotate_xticks=False,
    figsize=(10, 5),
):
    if y_col not in df_metrics.columns:
        return

    plot_df = df_metrics.sort_values(y_col).copy()

    plt.figure(figsize=figsize)
    sns.barplot(data=plot_df, x=x_col, y=y_col, color="steelblue")
    plt.axhline(plot_df[y_col].mean(), color="red", linestyle="--", label="Mean")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if rotate_xticks:
        plt.xticks(rotation=90)

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_true_vs_pred_scatter(
    df_predictions,
    key_output,
    out_dir,
    file_stub,
    lag_feature=False,
    figsize=(6, 6),
):
    plt.figure(figsize=figsize)

    x_true = df_predictions[f"true_{key_output}"]
    y_pred = df_predictions[f"pred_{key_output}"]

    plt.scatter(x_true, y_pred, alpha=0.25)

    lims = [
        min(x_true.min(), y_pred.min()),
        max(x_true.max(), y_pred.max())
    ]
    plt.plot(
        lims,
        lims,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="True = Predicted"
    )

    plt.xlabel(f"True {key_output}")
    plt.ylabel(f"Predicted {key_output}")
    plt.title(f"True vs predicted {key_output}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_outputs_true_vs_pred(
    df_predictions,
    target_cols,
    out_dir,
    file_stub,
    lag_feature=False,
    figsize=(8, 8),
):
    plt.figure(figsize=figsize)

    all_true = []
    all_pred = []

    for col in target_cols:
        if f"true_{col}" not in df_predictions.columns or f"pred_{col}" not in df_predictions.columns:
            continue

        style = OUTPUT_STYLES.get(col, {"color": None, "marker": "o", "label": col})

        x_true = df_predictions[f"true_{col}"].to_numpy()
        y_pred = df_predictions[f"pred_{col}"].to_numpy()

        mask = np.isfinite(x_true) & np.isfinite(y_pred)
        x_true = x_true[mask]
        y_pred = y_pred[mask]

        if len(x_true) == 0:
            continue

        all_true.extend(x_true.tolist())
        all_pred.extend(y_pred.tolist())

        plt.scatter(
            x_true,
            y_pred,
            color=style["color"],
            marker=style["marker"],
            alpha=0.35,
            label=style["label"],
        )

    if len(all_true) == 0 or len(all_pred) == 0:
        plt.close()
        return

    lim_min = min(np.nanmin(all_true), np.nanmin(all_pred))
    lim_max = max(np.nanmax(all_true), np.nanmax(all_pred))

    plt.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="True = Predicted",
    )

    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title("True vs predicted for all outputs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_example_trace(
    example_df,
    key_output,
    time_col,
    title,
    out_dir,
    file_stub,
    lag_feature=False,
    figsize=(12, 4),
):
    plt.figure(figsize=figsize)

    plt.plot(
        example_df[time_col],
        example_df[f"true_{key_output}"],
        linestyle="-",
        linewidth=2.0,
        label=f"True {key_output}",
    )
    plt.plot(
        example_df[time_col],
        example_df[f"pred_{key_output}"],
        linestyle="--",
        linewidth=2.0,
        alpha=0.95,
        label=f"Predicted {key_output}",
    )

    plt.xlabel("Time (s)")
    plt.ylabel(key_output)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_outputs_time_traces(
    example_df,
    x_col,
    target_cols,
    title,
    out_dir,
    file_stub,
    lag_feature=False,
    figsize=(14, 8),
):
    n_outputs = len(target_cols)
    ncols = 2
    nrows = int(np.ceil(n_outputs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, target_cols):
        if f"true_{col}" not in example_df.columns or f"pred_{col}" not in example_df.columns:
            ax.axis("off")
            continue

        style = OUTPUT_STYLES.get(col, {"color": None, "label": col})

        ax.plot(
            example_df[x_col],
            example_df[f"true_{col}"],
            color=style["color"],
            linestyle="-",
            linewidth=2.0,
            label=f"True {style['label']}",
        )
        ax.plot(
            example_df[x_col],
            example_df[f"pred_{col}"],
            color=style["color"],
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            label=f"Predicted {style['label']}",
        )

        ax.set_title(style["label"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(col)
        ax.legend()

    for ax in axes[n_outputs:]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_feature_importance_bar(
    df_importance,
    title,
    out_dir,
    file_stub,
    lag_feature=False,
    topn=10,
    figsize=(8, 5),
):
    topn = min(topn, len(df_importance))
    df_top = df_importance.head(topn).sort_values("importance")

    plt.figure(figsize=figsize)
    plt.barh(df_top["feature"], df_top["importance"], color="darkgreen")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    '''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT_STYLES = {
    "glideslope_error_deg": {"color": "tab:blue", "marker": "x", "label": "Glideslope error"},
    "localizer_error_deg": {"color": "goldenrod", "marker": "s", "label": "Localizer error"},
    "airspeed_error_kts": {"color": "tab:red", "marker": "D", "label": "Airspeed error"},
    "total_error": {"color": "tab:green", "marker": "o", "label": "Total error"},
}


def save_name(base_name, lag_feature=False, ext="png"):
    if lag_feature:
        return f"{base_name}_lag.{ext}"
    return f"{base_name}.{ext}"


def get_output_style(output_name):
    return OUTPUT_STYLES.get(
        output_name,
        {"color": None, "marker": "o", "label": output_name}
    )


def plot_rmse_bar(
    df_metrics,
    x_col,
    y_col,
    x_label,
    y_label,
    title,
    out_dir,
    file_stub,
    lag_feature=False,
    rotate_xticks=False,
    figsize=(10, 5),
):
    if y_col not in df_metrics.columns:
        return

    plot_df = df_metrics.sort_values(y_col).copy()

    plt.figure(figsize=figsize)
    sns.barplot(data=plot_df, x=x_col, y=y_col, color="steelblue")
    plt.axhline(plot_df[y_col].mean(), color="red", linestyle="--", label="Mean")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if rotate_xticks:
        plt.xticks(rotation=90)

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_true_vs_pred_scatter(
    df_predictions,
    key_output,
    out_dir,
    file_stub,
    lag_feature=False,
    figsize=(6, 6),
):
    plt.figure(figsize=figsize)

    style = get_output_style(key_output)
    x_true = df_predictions[f"true_{key_output}"]
    y_pred = df_predictions[f"pred_{key_output}"]

    plt.scatter(
        x_true,
        y_pred,
        alpha=0.35,
        color=style["color"],
        marker=style["marker"],
        label=style["label"],
    )

    lims = [
        min(x_true.min(), y_pred.min()),
        max(x_true.max(), y_pred.max())
    ]
    plt.plot(
        lims,
        lims,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="True = Predicted"
    )

    plt.xlabel(f"True {key_output}")
    plt.ylabel(f"Predicted {key_output}")
    plt.title(f"True vs predicted {key_output}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_example_trace(
    example_df,
    key_output,
    time_col,
    title,
    out_dir,
    file_stub,
    lag_feature=False,
    figsize=(12, 4),
):
    style = get_output_style(key_output)

    plt.figure(figsize=figsize)
    plt.plot(
        example_df[time_col],
        example_df[f"true_{key_output}"],
        color=style["color"],
        linestyle="-",
        linewidth=2.0,
        label=f"True {style['label']}",
    )
    plt.plot(
        example_df[time_col],
        example_df[f"pred_{key_output}"],
        color=style["color"],
        linestyle="--",
        linewidth=2.0,
        alpha=0.95,
        label=f"Predicted {style['label']}",
    )
    plt.xlabel("Time (s)")
    plt.ylabel(key_output)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_outputs_time_traces(
    example_df,
    x_col,
    target_cols,
    title,
    out_dir,
    file_stub,
    lag_feature=False,
    figsize=(14, 8),
):
    n_outputs = len(target_cols)
    ncols = 2
    nrows = int(np.ceil(n_outputs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, target_cols):
        if f"true_{col}" not in example_df.columns or f"pred_{col}" not in example_df.columns:
            ax.axis("off")
            continue

        style = get_output_style(col)

        ax.plot(
            example_df[x_col],
            example_df[f"true_{col}"],
            color=style["color"],
            linestyle="-",
            linewidth=2.0,
            label=f"True {style['label']}",
        )
        ax.plot(
            example_df[x_col],
            example_df[f"pred_{col}"],
            color=style["color"],
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            label=f"Predicted {style['label']}",
        )
        ax.set_title(style["label"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(col)
        ax.legend()

    for ax in axes[n_outputs:]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_feature_importance_bar(
    df_importance,
    title,
    out_dir,
    file_stub,
    lag_feature=False,
    topn=10,
    figsize=(8, 5),
):
    topn = min(topn, len(df_importance))
    df_top = df_importance.head(topn).sort_values("importance")

    plt.figure(figsize=figsize)
    plt.barh(df_top["feature"], df_top["importance"], color="darkgreen")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

'''
def compute_error_correlations(
    df_predictions,
    target_cols,
    total_error_col="total_error",
):
    rows = []

    true_total_col = f"true_{total_error_col}"
    if true_total_col not in df_predictions.columns:
        return pd.DataFrame(columns=[
            "output", "series_type", "x_col", "y_col", "pearson_r", "n"
        ])

    for col in target_cols:
        #if col == total_error_col:
            #continue

        for series_type, y_col in [
            ("true", f"true_{col}"),
            ("pred", f"pred_{col}"),
        ]:
            if y_col not in df_predictions.columns:
                continue

            tmp = df_predictions[[true_total_col, y_col]].dropna().copy()
            if len(tmp) < 2:
                r_val = np.nan
            else:
                r_val = tmp[true_total_col].corr(tmp[y_col], method="pearson")

            rows.append({
                "output": col,
                "series_type": series_type,
                "x_col": true_total_col,
                "y_col": y_col,
                "pearson_r": r_val,
                "n": len(tmp),
            })

    df_corr = pd.DataFrame(rows)
    return df_corr
'''

def compute_error_correlations(
    df_predictions,
    target_cols,
    total_error_col="total_error",
):
    rows = []

    true_total_col = f"true_{total_error_col}"
    if true_total_col not in df_predictions.columns:
        return pd.DataFrame(columns=[
            "output", "series_type", "x_col", "y_col", "pearson_r", "n"
        ])

    for col in target_cols:
        for series_type, y_col in [
            ("true", f"true_{col}"),
            ("pred", f"pred_{col}"),
        ]:
            if y_col not in df_predictions.columns:
                continue

            if y_col == true_total_col:
                x = pd.to_numeric(
                    df_predictions.loc[:, true_total_col],
                    errors="coerce"
                ).dropna().to_numpy().ravel()
                n_val = len(x)
                r_val = 1.0 if n_val >= 2 else np.nan
            else:
                tmp = df_predictions.loc[:, [true_total_col, y_col]].copy()
                tmp[true_total_col] = pd.to_numeric(tmp[true_total_col], errors="coerce")
                tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
                tmp = tmp.dropna()

                x = tmp.loc[:, true_total_col].to_numpy().ravel()
                y = tmp.loc[:, y_col].to_numpy().ravel()
                n_val = len(tmp)

                if n_val < 2:
                    r_val = np.nan
                else:
                    r_val = np.corrcoef(x, y)[0, 1]

            rows.append({
                "output": col,
                "series_type": series_type,
                "x_col": true_total_col,
                "y_col": y_col,
                "pearson_r": r_val,
                "n": n_val,
            })

    return pd.DataFrame(rows)


def save_error_correlations_csv(
    df_predictions,
    target_cols,
    out_dir,
    file_stub="error_correlations_vs_true_total_error",
    lag_feature=False,
    total_error_col="total_error",
):
    df_corr = compute_error_correlations(
        df_predictions=df_predictions,
        target_cols=target_cols,
        total_error_col=total_error_col,
    )
    df_corr.to_csv(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "csv")),
        index=False
    )
    return df_corr

'''
def plot_correlations_vs_true_total_error(
    df_predictions,
    target_cols,
    out_dir,
    file_stub="corr_vs_true_total_error",
    lag_feature=False,
    total_error_col="total_error",
    figsize=(11, 7),
):
    true_total_col = f"true_{total_error_col}"
    if true_total_col not in df_predictions.columns:
        return

    plt.figure(figsize=figsize)

    for col in target_cols:
        #if col == total_error_col:
            #continue

        style = get_output_style(col)

        for series_type, y_col, line_style, alpha in [
            ("true", f"true_{col}", "-", 0.9),
            ("pred", f"pred_{col}", "--", 0.75),
        ]:
            if y_col not in df_predictions.columns:
                continue

            tmp = df_predictions[[true_total_col, y_col]].dropna().copy()
            if len(tmp) == 0:
                continue

            x = tmp[true_total_col].to_numpy()
            y = tmp[y_col].to_numpy()

            plt.scatter(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                alpha=0.35 if series_type == "pred" else 0.5,
                s=18,
                label=f"{'True' if series_type == 'true' else 'Pred'} {style['label']}",
            )

            if len(tmp) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(np.min(x), np.max(x), 100)
                y_line = slope * x_line + intercept
                r_val = np.corrcoef(x, y)[0, 1]

                plt.plot(
                    x_line,
                    y_line,
                    color=style["color"],
                    linestyle=line_style,
                    linewidth=1.5,
                    alpha=alpha,
                    label=f"r = {r_val:.2f}",
                )

    plt.xlabel("True Total Error")
    plt.ylabel("Value")
    plt.title("Correlations with true total error")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    '''

def plot_correlations_vs_true_total_error(
    df_predictions,
    target_cols,
    out_dir,
    file_stub="corr_vs_true_total_error",
    lag_feature=False,
    total_error_col="total_error",
    figsize=(11, 7),
):
    true_total_col = f"true_{total_error_col}"
    if true_total_col not in df_predictions.columns:
        return

    plt.figure(figsize=figsize)
    legend_added = set()

    for col in target_cols:
        style = get_output_style(col)

        for series_type, y_col, line_style, alpha in [
            ("true", f"true_{col}", "-", 0.95),
            ("pred", f"pred_{col}", "--", 0.75),
        ]:
            if y_col not in df_predictions.columns:
                continue

            if y_col == true_total_col:
                x = pd.to_numeric(
                    df_predictions.loc[:, true_total_col],
                    errors="coerce"
                ).dropna().to_numpy().ravel()
                y = x.copy()
            else:
                tmp = df_predictions.loc[:, [true_total_col, y_col]].copy()
                tmp[true_total_col] = pd.to_numeric(tmp[true_total_col], errors="coerce")
                tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
                tmp = tmp.dropna()

                if len(tmp) == 0:
                    continue

                x = tmp.loc[:, true_total_col].to_numpy().ravel()
                y = tmp.loc[:, y_col].to_numpy().ravel()

            if len(x) < 2 or len(y) < 2:
                continue

            point_label = f"{'True' if series_type == 'true' else 'Pred'} {style['label']}"
            if point_label not in legend_added:
                plt.scatter(
                    x,
                    y,
                    color=style["color"],
                    marker=style["marker"],
                    alpha=0.35 if series_type == "pred" else 0.5,
                    s=18,
                    label=point_label,
                )
                legend_added.add(point_label)
            else:
                plt.scatter(
                    x,
                    y,
                    color=style["color"],
                    marker=style["marker"],
                    alpha=0.35 if series_type == "pred" else 0.5,
                    s=18,
                )

            if col == total_error_col and series_type == "true":
                x_line = np.linspace(np.min(x), np.max(x), 100)
                y_line = x_line.copy()
                r_val = 1.0
            else:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(np.min(x), np.max(x), 100)
                y_line = slope * x_line + intercept
                r_val = np.corrcoef(x, y)[0, 1]

            plt.plot(
                x_line,
                y_line,
                color=style["color"],
                linestyle=line_style,
                linewidth=1.5,
                alpha=alpha,
                label=f"{'True' if series_type == 'true' else 'Pred'} {style['label']} r = {r_val:.2f}",
            )

    plt.xlabel("True Total Error")
    plt.ylabel("Value")
    plt.title("Correlations with true total error")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, save_name(file_stub, lag_feature, "png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()