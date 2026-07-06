"""
data_analysis.py — smolagents CodeAgent that explores uploaded CSV/XLSX
files, builds charts, and writes a Markdown report. The agent can install
any missing Python packages itself via the `install_package` tool.
"""

import inspect
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from smolagents import CodeAgent, tool

import agent_memory
import model_registry as mr
import models
from hardware import DEVICE
from knowledge_base import get_file_path

_data_agent          = None
_data_agent_model_id = None
_data_agent_lock      = threading.Lock()

# How many conversation turns the Data Analysis CodeAgent is allowed to
# remember (see agent_memory.cap_agent_memory()). Kept smaller than the
# other tabs' default of 6 because each turn's task prompt here is already
# long (the full EDA instructions in run_data_analysis() below), so memory
# grows the prompt much faster per turn than a one-line chat question would.
DATA_AGENT_MEMORY_TURNS = 3

# Tracks which uploaded file paths the agent's current memory "belongs
# to". A new/changed set of files means a different dataset, so old
# memory (which may reference the previous file's columns/stats) needs to
# be dropped even though the model itself hasn't changed — see
# agent_memory.reset_if_context_changed().
_last_data_context = {"key": None}


@tool
def install_package(package_name: str) -> str:
    """
    Install a Python package into the current environment using pip.
    Use this whenever a data-analysis step needs a library that is not
    yet installed (e.g. "openpyxl" for reading .xlsx files, "seaborn",
    "scikit-learn", "statsmodels", "plotly", "xlsxwriter").

    Args:
        package_name: The pip package name to install, e.g. "seaborn" or
            "scikit-learn==1.4.0". Pass a single package per call.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", package_name],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return f"✅ Installed '{package_name}' successfully."
        return f"❌ Failed to install '{package_name}':\n{result.stderr[-2000:]}"
    except subprocess.TimeoutExpired:
        return f"❌ Installing '{package_name}' timed out after 300s."
    except Exception as e:
        return f"❌ Error installing '{package_name}': {e}"


@tool
def save_report(content: str, filename: str = "report.md") -> str:
    """
    Save Markdown report text to a file inside the data-analysis output
    directory.

    IMPORTANT: the sandboxed code executor blocks Python's built-in
    open()/write() for safety — calling open() directly always fails
    with "Forbidden function evaluation". Use THIS tool to save your
    report instead. (matplotlib's plt.savefig() is a separate whitelisted
    call and works fine for charts — no change needed there.)

    Args:
        content: The full Markdown report text to save.
        filename: File name to save it as, e.g. "report.md" (default).
    """
    try:
        out_dir = Path(mr.DATA_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        path.write_text(content, encoding="utf-8")
        return f"✅ Saved report to '{path}' ({len(content)} chars)."
    except Exception as e:
        return f"❌ Failed to save report: {e}"


def get_data_agent(model_id: Optional[str] = None):
    """Lazily build (or rebuild, if the model changed) the data-analysis CodeAgent."""
    global _data_agent, _data_agent_model_id
    target = model_id or models._llm_model_id

    if _data_agent is not None and target == _data_agent_model_id:
        return _data_agent

    with _data_agent_lock:
        if _data_agent is not None and target == _data_agent_model_id:
            return _data_agent

        print(f"[DataAgent] Building CodeAgent on '{target}' …")
        llm = models.get_llm(target)
        agent_kwargs = dict(
            model=llm,
            tools=[install_package, save_report],
            additional_authorized_imports=["*"],   # trusted local machine — full stdlib + installed pkgs
            max_steps=mr.DATA_AGENT_MAX_STEPS,
        )
        # Some models (esp. "thinking"-tuned ones, or anything running
        # through a raw llama.cpp chat template) reliably write plain
        # ```python fenced code instead of the <code></code> tags
        # CodeAgent expects by default, causing every step to fail
        # parsing. Use the more broadly-compatible markdown-fence
        # convention when this smolagents version supports it.
        try:
            if "code_block_tags" in inspect.signature(CodeAgent.__init__).parameters:
                agent_kwargs["code_block_tags"] = "markdown"
        except (TypeError, ValueError):
            pass
        _data_agent = CodeAgent(**agent_kwargs)
        _data_agent_model_id = target
        # Standard smolagents behaviour: a freshly-built CodeAgent starts
        # with empty memory. This only happens on first use, or after a
        # model switch/reset via reset_agent() below — dataset changes are
        # handled separately by reset_if_context_changed() in
        # run_data_analysis(), which clears an EXISTING agent's memory in
        # place rather than rebuilding it (see agent_memory.py's module
        # docstring for why rebuilding-to-reset used to reintroduce the
        # very stale memory it was meant to drop).
        return _data_agent


def reset_agent():
    """Drop the CodeAgent wrapper. It references the shared LLM managed by
    models.get_llm(), whose memory is released there when the underlying
    model actually changes. Rebuilding is cheap since it reuses get_llm(target).

    Also called whenever the LLM is reloaded/unloaded elsewhere (the agent
    holds its own reference to that same model and would otherwise keep an
    unloaded model alive).
    """
    global _data_agent, _data_agent_model_id
    _data_agent = None
    _data_agent_model_id = None


def save_data_files(files) -> list:
    """Copy uploaded CSV/XLSX files into the persistent data-analysis workspace."""
    Path(mr.DATA_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    paths = []
    if not files:
        return paths
    if not isinstance(files, list):
        files = [files]
    for f in files:
        src = get_file_path(f)
        if not src:
            continue
        dest = Path(mr.DATA_UPLOAD_DIR) / Path(src).name
        try:
            shutil.copy(src, dest)
            paths.append(str(dest))
        except Exception:
            pass
    return paths


def run_data_analysis(files, question: str, model_label: str, history: list, use_memory: bool = True):
    """Hand uploaded data + the user's question to the CodeAgent and collect its report.

    `use_memory` controls the "🧠 Conversation Memory" checkbox: when on,
    the CodeAgent remembers earlier turns about the SAME dataset (capped —
    see DATA_AGENT_MEMORY_TURNS) and old memory is dropped automatically
    if a different file is uploaded. When off, every question is answered
    from a clean slate regardless of what file is uploaded.
    """
    history = history or []

    paths = save_data_files(files)
    if not paths:
        history.append({"role": "user", "content": question or "(no file)"})
        history.append({"role": "assistant", "content": "⚠️ Please upload a CSV or XLSX file first."})
        return history, None, None

    question = (question or "").strip() or (
        "Perform a full exploratory data analysis (EDA) on this dataset and "
        "generate a thorough Markdown report with multiple charts."
    )
    model_id = mr.MODEL_OPTIONS.get(model_label, mr.DEFAULT_LLM_MODEL)
    history.append({"role": "user", "content": f"📎 {', '.join(Path(p).name for p in paths)}\n\n{question}"})

    try:
        out_dir = Path(mr.DATA_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Now that a run can produce many charts (full EDA, not just one),
        # clear out PNGs left over from a previous analysis first — otherwise
        # they'd pile up and get mixed into this run's gallery below.
        for stale_png in out_dir.glob("*.png"):
            try:
                stale_png.unlink()
            except OSError:
                pass
        agent = get_data_agent(model_id)
        # New/changed dataset → old agent memory (referencing whatever the
        # previous file's columns/stats were) is no longer relevant, so
        # drop it even though the model itself hasn't changed. Only
        # relevant when memory is on — when it's off, reset=True below
        # already makes every run stateless regardless of dataset.
        # Clears agent.memory.steps directly on the SAME agent object —
        # see agent_memory.reset_if_context_changed()/reset_memory()'s
        # docstrings for why this (rather than tearing down and rebuilding
        # the agent, as an earlier version of this file did) is required
        # for the reset to actually stick.
        if use_memory:
            agent_memory.reset_if_context_changed(agent, _last_data_context, tuple(sorted(paths)))
        file_list_str = "\n".join(f"- {p}" for p in paths)
        report_path = str(Path(mr.DATA_OUTPUT_DIR) / "report.md")

        task = f"""You are a data analysis assistant performing a thorough Exploratory
Data Analysis (EDA) with pandas and matplotlib, in a local Python sandbox.

Data file(s) provided by the user:
{file_list_str}

User request: {question}

Do a REAL exploratory analysis, not just a one-paragraph summary. Work through
every relevant section below (skip a section only if it genuinely doesn't apply —
e.g. no categorical columns exist, or only one numeric column exists so no
correlation is possible). Aim for several charts, not just one.

1. LOAD & OVERVIEW
   - Load the file(s) with pandas (pd.read_csv for .csv, pd.read_excel for
     .xlsx/.xls — if a required package like 'openpyxl' is missing, call the
     install_package tool with its pip name first, then retry the import).
   - Report shape, column names, dtypes, memory usage, missing-value counts
     (and %), and duplicate-row count.
   - Split columns into numeric vs categorical (use pd.api.types.is_numeric_dtype
     / is_datetime64_any_dtype — do NOT use nonexistent methods like
     select_d_dtype; the real pandas method is data.select_dtypes(include=...)).

2. UNIVARIATE ANALYSIS (per numeric column, or the most important few if there
   are many)
   - Descriptive statistics: mean, median, std, min, max, quartiles, skewness.
   - For EACH numeric column (or top 5 most relevant if there are more), plot a
     histogram AND a boxplot (either as two separate PNGs, or combined into one
     figure with subplots) to show distribution shape and outliers.
   - For each categorical column (or top 5 most relevant), plot a bar chart of
     value counts (group rare categories into "Other" if there are more than
     ~10 distinct values).
   - Note any skew, outliers, or unusual patterns you observe in the text report.

3. BIVARIATE / RELATIONSHIP ANALYSIS
   - If there are 2+ numeric columns: compute the correlation matrix
     (data.corr(numeric_only=True)) and plot it as a heatmap using
     matplotlib's plt.imshow(...) with a colorbar and axis tick labels (no
     seaborn required, but you may install_package("seaborn") and use it if
     you prefer — either is fine).
   - Pick the 1-3 most correlated (or most business-relevant, based on the
     user's request) numeric column pairs and make scatter plots of each pair.
   - If there's an obvious categorical grouping column, make at least one
     grouped comparison chart (e.g. bar chart of a numeric column's mean per
     category, or overlaid/side-by-side boxplots per category).

4. OUTLIERS
   - Flag outliers using the IQR method (values beyond Q1 - 1.5*IQR or
     Q3 + 1.5*IQR) for at least the 1-2 most important numeric columns, and
     report how many outlier rows were found per column.

5. SAVE EVERY CHART
   - Save every chart as its own PNG with a descriptive filename inside the
     directory '{mr.DATA_OUTPUT_DIR}' (use plt.savefig(...); ALWAYS call
     plt.close() right after savefig so figures don't bleed into each other;
     never call plt.show()).

6. WRITE THE REPORT
   - Write a well-structured Markdown report with headings for each section
     above (Overview, Univariate Analysis, Correlation/Relationships,
     Outliers, Key Insights), including the actual numbers you computed (not
     placeholders) and 3-5 concrete bullet-point insights/observations at the
     end — not just restating the stats, but what they mean (e.g. "X is
     right-skewed with several high outliers", "A and B are strongly
     correlated (r=0.82), suggesting...").
   - Reference the chart filenames you created in the relevant sections so a
     reader knows which chart supports which point.
   - Call the save_report tool with that Markdown text to save it, e.g.:
     save_report(content=report_text). Do NOT use Python's open()/write() to
     save the report — the sandbox blocks raw open() and that call always
     fails. (plt.savefig() is a separate whitelisted call and works fine for
     charts — only raw open() is blocked.)

7. As your FINAL ANSWER, return the full Markdown report text.
"""
        t0 = time.time()
        # reset=False keeps this CodeAgent's memory across turns on the
        # SAME dataset (e.g. "now also break that down by region" after an
        # initial EDA) — see
        # https://huggingface.co/docs/smolagents/tutorials/memory. Capped
        # right after via agent_memory.cap_agent_memory(), and fully reset
        # above whenever the uploaded file(s) change. reset=True (memory
        # checkbox off) makes every question stateless instead.
        result = agent.run(task, reset=not use_memory)
        if use_memory:
            agent_memory.cap_agent_memory(agent, max_turns=DATA_AGENT_MEMORY_TURNS)
        elapsed = time.time() - t0

        report_text = str(result)
        response = (
            report_text +
            f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> "
            f"({DEVICE.upper()}) | memory: {'on' if use_memory else 'off'}</sub>"
        )
        history.append({"role": "assistant", "content": response})

        chart_files = sorted(str(p) for p in Path(mr.DATA_OUTPUT_DIR).glob("*.png"))
        report_file = report_path if Path(report_path).exists() else None
        return history, (chart_files or None), report_file

    except Exception as e:
        import traceback
        history.append({"role": "assistant", "content": f"❌ {e}\n\n{traceback.format_exc()}"})
        return history, None, None
