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


def sample_for_plot(df, max_points=5000, random_state=42):
    if max_points is None or len(df) <= max_points:
        return df.copy()
    return df.sample(n=max_points, random_state=random_state).copy()


def stratified_sample_for_plot(
    df,
    stratify_col,
    max_points=6000,
    n_bins=20,
    random_state=42,
):
    if max_points is None or len(df) <= max_points:
        return df.copy()

    tmp = df.copy()
    tmp = tmp.dropna(subset=[stratify_col])
    if len(tmp) <= max_points:
        return tmp

    n_unique = tmp[stratify_col].nunique()
    if n_unique < 2:
        return tmp.sample(n=min(max_points, len(tmp)), random_state=random_state).copy()

    tmp["_plot_bin"] = pd.qcut(
        tmp[stratify_col],
        q=min(n_bins, n_unique),
        duplicates="drop"
    )

    groups = list(tmp.groupby("_plot_bin", observed=False))
    if len(groups) == 0:
        return tmp.sample(n=max_points, random_state=random_state).copy()

    per_bin = max(1, max_points // len(groups))
    sampled = []

    for _, df_bin in groups:
        take_n = min(per_bin, len(df_bin))
        sampled.append(df_bin.sample(n=take_n, random_state=random_state))

    df_plot = pd.concat(sampled, axis=0).copy()

    if len(df_plot) > max_points:
        df_plot = df_plot.sample(n=max_points, random_state=random_state).copy()

    return df_plot.drop(columns="_plot_bin", errors="ignore")


def downsample_timeseries(df, max_points=1500):
    if max_points is None or len(df) <= max_points:
        return df.copy()

    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step, :].copy()


def make_binned_summary(df, x_col, y_col, n_bins=20):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    tmp = pd.DataFrame({
        "x_val": x,
        "y_val": y,
    }).dropna()

    if len(tmp) == 0:
        return pd.DataFrame(columns=["x_mean", "y_mean", "y_std", "n"])

    n_unique = tmp["x_val"].nunique()
    if n_unique < 2:
        return pd.DataFrame({
            "x_mean": [tmp["x_val"].mean()],
            "y_mean": [tmp["y_val"].mean()],
            "y_std": [tmp["y_val"].std()],
            "n": [len(tmp)],
        })

    tmp["_bin"] = pd.qcut(
        tmp["x_val"],
        q=min(n_bins, n_unique),
        duplicates="drop"
    )

    df_bin = (
        tmp.groupby("_bin", observed=False)
        .agg(
            x_mean=("x_val", "mean"),
            y_mean=("y_val", "mean"),
            y_std=("y_val", "std"),
            n=("y_val", "size"),
        )
        .reset_index(drop=True)
    )

    return df_bin


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
    max_points=5000,
):
    true_col = f"true_{key_output}"
    pred_col = f"pred_{key_output}"
    if true_col not in df_predictions.columns or pred_col not in df_predictions.columns:
        return

    df_plot = sample_for_plot(df_predictions[[true_col, pred_col]], max_points=max_points, random_state=42)

    plt.figure(figsize=figsize)

    style = get_output_style(key_output)
    x_true = pd.to_numeric(df_plot[true_col], errors="coerce")
    y_pred = pd.to_numeric(df_plot[pred_col], errors="coerce")

    mask = x_true.notna() & y_pred.notna()
    x_true = x_true[mask]
    y_pred = y_pred[mask]

    plt.scatter(
        x_true,
        y_pred,
        alpha=0.30,
        color=style["color"],
        marker=style["marker"],
        s=24,
        label=f"{style['label']} (n={len(x_true)})",
    )

    if len(x_true) > 0:
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
    max_points=1500,
):
    true_col = f"true_{key_output}"
    pred_col = f"pred_{key_output}"
    if true_col not in example_df.columns or pred_col not in example_df.columns or time_col not in example_df.columns:
        return

    style = get_output_style(key_output)
    df_plot = downsample_timeseries(example_df.sort_values(time_col), max_points=max_points)

    plt.figure(figsize=figsize)
    plt.plot(
        df_plot[time_col],
        df_plot[true_col],
        color=style["color"],
        linestyle="-",
        linewidth=2.0,
        label=f"True {style['label']}",
    )
    plt.plot(
        df_plot[time_col],
        df_plot[pred_col],
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
    max_points=1500,
):
    if x_col not in example_df.columns:
        return

    df_plot = downsample_timeseries(example_df.sort_values(x_col), max_points=max_points)

    n_outputs = len(target_cols)
    ncols = 2
    nrows = int(np.ceil(n_outputs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, target_cols):
        true_col = f"true_{col}"
        pred_col = f"pred_{col}"

        if true_col not in df_plot.columns or pred_col not in df_plot.columns:
            ax.axis("off")
            continue

        style = get_output_style(col)

        ax.plot(
            df_plot[x_col],
            df_plot[true_col],
            color=style["color"],
            linestyle="-",
            linewidth=2.0,
            label=f"True {style['label']}",
        )
        ax.plot(
            df_plot[x_col],
            df_plot[pred_col],
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

            if col == total_error_col and series_type == "true":
                tmp = df_predictions[[true_total_col]].dropna().copy()
                r_val = 1.0 if len(tmp) >= 2 else np.nan
                n_val = len(tmp)
            else:
                tmp = df_predictions[[true_total_col, y_col]].dropna().copy()
                if len(tmp) < 2:
                    r_val = np.nan
                else:
                    r_val = tmp[true_total_col].corr(tmp[y_col], method="pearson")
                n_val = len(tmp)

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


def plot_correlations_vs_true_total_error(
    df_predictions,
    target_cols,
    out_dir,
    file_stub="corr_vs_true_total_error",
    lag_feature=False,
    total_error_col="total_error",
    figsize=(11, 7),
    n_bins=20,
):
    true_total_col = f"true_{total_error_col}"
    if true_total_col not in df_predictions.columns:
        return

    plt.figure(figsize=figsize)
    marker_labels_added = set()
    line_labels_added = set()

    for col in target_cols:
        style = get_output_style(col)

        for series_type, y_col, line_style, line_alpha in [
            ("true", f"true_{col}", "-", 0.95),
            ("pred", f"pred_{col}", "--", 0.80),
        ]:
            if y_col not in df_predictions.columns:
                continue

            if col == total_error_col and series_type == "true":
                full_df = df_predictions[[true_total_col]].copy()
                full_df[true_total_col] = pd.to_numeric(full_df[true_total_col], errors="coerce")
                full_df = full_df.dropna()

                if len(full_df) < 2:
                    continue

                x_full = full_df[true_total_col].to_numpy().ravel()
                y_full = x_full.copy()

                tmp_for_bin = pd.DataFrame({
                    "x_val": x_full,
                    "y_val": y_full,
                })
                df_bin = make_binned_summary(
                    tmp_for_bin,
                    x_col="x_val",
                    y_col="y_val",
                    n_bins=n_bins,
                )
            else:
                full_df = df_predictions[[true_total_col, y_col]].copy()
                full_df[true_total_col] = pd.to_numeric(full_df[true_total_col], errors="coerce")
                full_df[y_col] = pd.to_numeric(full_df[y_col], errors="coerce")
                full_df = full_df.dropna()

                if len(full_df) < 2:
                    continue

                x_full = full_df[true_total_col].to_numpy().ravel()
                y_full = full_df[y_col].to_numpy().ravel()

                df_bin = make_binned_summary(
                    full_df,
                    x_col=true_total_col,
                    y_col=y_col,
                    n_bins=n_bins,
                )

            label_base = f"{'True' if series_type == 'true' else 'Pred'} {style['label']}"

            # binned markers only, no connecting line
            if len(df_bin) > 0:
                if series_type == "true":
                    marker_kwargs = dict(
                        linestyle="None",
                        marker="o",
                        markersize=5.5,
                        markerfacecolor="none",
                        markeredgecolor=style["color"],
                        markeredgewidth=1.2,
                        color=style["color"],
                        alpha=0.95,
                    )
                else:
                    marker_kwargs = dict(
                        linestyle="None",
                        marker="x",
                        markersize=5.5,
                        color=style["color"],
                        markeredgewidth=1.2,
                        alpha=0.90,
                    )

                if label_base not in marker_labels_added:
                    plt.plot(
                        df_bin["x_mean"],
                        df_bin["y_mean"],
                        # linestyle="None",
                        # marker=style["marker"],
                        # markersize=5,
                        # color=style["color"],
                        # alpha=0.95 if series_type == "true" else 0.85,
                        label=label_base,
                        **marker_kwargs   
                    )
                    marker_labels_added.add(label_base)
                else:
                    plt.plot(
                        df_bin["x_mean"],
                        df_bin["y_mean"],
                        # linestyle="None",
                        # marker=style["marker"],
                        # markersize=5,
                        # color=style["color"],
                        # alpha=0.95 if series_type == "true" else 0.85,
                        **marker_kwargs
                    )

            # regression line from full raw data
            if col == total_error_col and series_type == "true":
                x_line = np.linspace(np.min(x_full), np.max(x_full), 100)
                y_line = x_line.copy()
                r_val = 1.0
            else:
                slope, intercept = np.polyfit(x_full, y_full, 1)
                x_line = np.linspace(np.min(x_full), np.max(x_full), 100)
                y_line = slope * x_line + intercept
                r_val = np.corrcoef(x_full, y_full)[0, 1]

            line_label = f"r = {r_val:.2f}"
            if line_label not in line_labels_added:
                plt.plot(
                    x_line,
                    y_line,
                    color=style["color"],
                    linestyle=line_style,
                    linewidth=1.4,
                    alpha=line_alpha,
                    label=line_label,
                )
                line_labels_added.add(line_label)
            else:
                plt.plot(
                    x_line,
                    y_line,
                    color=style["color"],
                    linestyle=line_style,
                    linewidth=1.4,
                    alpha=line_alpha,
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
    max_points_per_series=1200,
    n_bins=20,
    show_points=False,
):
    true_total_col = f"true_{total_error_col}"
    if true_total_col not in df_predictions.columns:
        return

    plt.figure(figsize=figsize)
    legend_added = set()

    for col in target_cols:
        style = get_output_style(col)

        for series_type, y_col, line_style, line_alpha in [
            ("true", f"true_{col}", "-", 0.95),
            ("pred", f"pred_{col}", "--", 0.75),
        ]:
            if y_col not in df_predictions.columns:
                continue

            if col == total_error_col and series_type == "true":
                full_df = df_predictions.loc[:, [true_total_col]].copy()
                full_df[true_total_col] = pd.to_numeric(full_df[true_total_col], errors="coerce")
                full_df = full_df.dropna()

                if len(full_df) < 2:
                    continue

                x_full = full_df[true_total_col].to_numpy().ravel()
                y_full = x_full.copy()

                plot_df = sample_for_plot(
                    full_df,
                    max_points=max_points_per_series,
                    random_state=42,
                )
                x_plot = plot_df[true_total_col].to_numpy().ravel()
                y_plot = x_plot.copy()

                tmp_for_bin = pd.DataFrame({
                    true_total_col: x_full,
                    y_col: y_full,
                })
                df_bin = make_binned_summary(
                    tmp_for_bin,
                    x_col=true_total_col,
                    y_col=y_col,
                    n_bins=n_bins,
                )
            else:
                full_df = df_predictions.loc[:, [true_total_col, y_col]].copy()
                full_df[true_total_col] = pd.to_numeric(full_df[true_total_col], errors="coerce")
                full_df[y_col] = pd.to_numeric(full_df[y_col], errors="coerce")
                full_df = full_df.dropna()

                if len(full_df) < 2:
                    continue

                x_full = full_df[true_total_col].to_numpy().ravel()
                y_full = full_df[y_col].to_numpy().ravel()

                plot_df = stratified_sample_for_plot(
                    full_df,
                    stratify_col=true_total_col,
                    max_points=max_points_per_series,
                    n_bins=n_bins,
                    random_state=42,
                )
                x_plot = plot_df[true_total_col].to_numpy().ravel()
                y_plot = plot_df[y_col].to_numpy().ravel()

                df_bin = make_binned_summary(
                    full_df,
                    x_col=true_total_col,
                    y_col=y_col,
                    n_bins=n_bins,
                )

            label_base = f"{'True' if series_type == 'true' else 'Pred'} {style['label']}"

            if show_points and len(x_plot) > 0:
                if f"{label_base}_points" not in legend_added:
                    plt.scatter(
                        x_plot,
                        y_plot,
                        color=style["color"],
                        marker=style["marker"],
                        alpha=0.10 if series_type == "pred" else 0.14,
                        s=8,
                        label=label_base,
                    )
                    legend_added.add(f"{label_base}_points")
                else:
                    plt.scatter(
                        x_plot,
                        y_plot,
                        color=style["color"],
                        marker=style["marker"],
                        alpha=0.10 if series_type == "pred" else 0.14,
                        s=8,
                    )

            if len(df_bin) > 0:
                if f"{label_base}_bin" not in legend_added:
                    plt.plot(
                        df_bin["x_mean"],
                        df_bin["y_mean"],
                        color=style["color"],
                        linestyle=line_style,
                        linewidth=2.2,
                        marker=style["marker"],
                        markersize=5,
                        label=label_base,
                    )
                    legend_added.add(f"{label_base}_bin")
                else:
                    plt.plot(
                        df_bin["x_mean"],
                        df_bin["y_mean"],
                        color=style["color"],
                        linestyle=line_style,
                        linewidth=2.2,
                        marker=style["marker"],
                        markersize=5,
                    )

            if col == total_error_col and series_type == "true":
                x_line = np.linspace(np.min(x_full), np.max(x_full), 100)
                y_line = x_line.copy()
                r_val = 1.0
            else:
                slope, intercept = np.polyfit(x_full, y_full, 1)
                x_line = np.linspace(np.min(x_full), np.max(x_full), 100)
                y_line = slope * x_line + intercept
                r_val = np.corrcoef(x_full, y_full)[0, 1]

            plt.plot(
                x_line,
                y_line,
                color=style["color"],
                linestyle=line_style,
                linewidth=1.3,
                alpha=line_alpha,
                label=f"{label_base} r = {r_val:.2f}",
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