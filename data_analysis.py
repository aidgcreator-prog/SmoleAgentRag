"""
data_analysis.py — smolagents CodeAgent that explores uploaded CSV/XLSX
files, builds charts, and writes a Markdown report. The agent can install
any missing Python packages itself via the `install_package` tool.
"""

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from smolagents import CodeAgent, tool

import model_registry as mr
import models
from hardware import DEVICE
from knowledge_base import get_file_path

_data_agent          = None
_data_agent_model_id = None
_data_agent_lock      = threading.Lock()


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
        _data_agent = CodeAgent(
            model=llm,
            tools=[install_package],
            additional_authorized_imports=["*"],   # trusted local machine — full stdlib + installed pkgs
            max_steps=mr.DATA_AGENT_MAX_STEPS,
        )
        _data_agent_model_id = target
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


def run_data_analysis(files, question: str, model_label: str, history: list):
    """Hand uploaded data + the user's question to the CodeAgent and collect its report."""
    history = history or []

    paths = save_data_files(files)
    if not paths:
        history.append({"role": "user", "content": question or "(no file)"})
        history.append({"role": "assistant", "content": "⚠️ Please upload a CSV or XLSX file first."})
        return history, None, None

    question = (question or "").strip() or (
        "Explore this dataset, summarize key statistics and trends, "
        "and generate a short report with at least one chart."
    )
    model_id = mr.MODEL_OPTIONS.get(model_label, mr.DEFAULT_LLM_MODEL)
    history.append({"role": "user", "content": f"📎 {', '.join(Path(p).name for p in paths)}\n\n{question}"})

    try:
        Path(mr.DATA_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        agent = get_data_agent(model_id)
        file_list_str = "\n".join(f"- {p}" for p in paths)
        report_path = str(Path(mr.DATA_OUTPUT_DIR) / "report.md")

        task = f"""You are a data analysis assistant working with pandas in a local Python sandbox.

Data file(s) provided by the user:
{file_list_str}

User request: {question}

Instructions:
1. Load the file(s) with pandas (pd.read_csv for .csv, pd.read_excel for .xlsx/.xls —
   if a required package like 'openpyxl' is missing, call the install_package tool with
   its pip name first, then retry the import).
2. Explore the data: shape, columns, dtypes, missing values, basic descriptive statistics.
3. Perform the analysis the user asked for. If a package you need isn't available,
   install it with the install_package tool instead of giving up.
4. Create at least one relevant chart with matplotlib and save each chart as a PNG file
   inside the directory '{mr.DATA_OUTPUT_DIR}' (use plt.savefig(...); do not call plt.show()).
5. Write a concise Markdown report of your findings (headings, bullet points, key numbers)
   and save it to '{report_path}'.
6. As your FINAL ANSWER, return the full Markdown report text.
"""
        t0 = time.time()
        result = agent.run(task)
        elapsed = time.time() - t0

        report_text = str(result)
        response = (
            report_text +
            f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> ({DEVICE.upper()})</sub>"
        )
        history.append({"role": "assistant", "content": response})

        chart_files = sorted(str(p) for p in Path(mr.DATA_OUTPUT_DIR).glob("*.png"))
        report_file = report_path if Path(report_path).exists() else None
        return history, (chart_files or None), report_file

    except Exception as e:
        import traceback
        history.append({"role": "assistant", "content": f"❌ {e}\n\n{traceback.format_exc()}"})
        return history, None, None
