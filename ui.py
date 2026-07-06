"""
ui.py — Builds the Gradio Blocks UI and wires every event handler.

Layout convention used on every tab that has settings: the primary input
control (message box, file upload, audio recorder) sits at the top of the
main (left) column, the chat/results area follows inside a `gr.Accordion`
that starts CLOSED and pops open automatically the first time there's
something to show — a reply came back, an upload finished, etc. Everything
secondary (model pickers, load/unload buttons, status messages) lives in a
narrow sidebar column to the right of the main column, instead of a
collapsed "⚙️ Settings" accordion stacked underneath the chat like before.

Tabs without a settings equivalent (Knowledge Base, About) keep their
original single-column layout.

All handlers are defined and wired at the top level of build_ui() (never
inside a gr.render() block) — see the "Gradio 6 breaks event handler
rebinding on render cycles" learning: defining .click()/.submit() inside a
@gr.render() causes all buttons/dropdowns to silently stop working after
the first language switch.

INPUT-BOX CLEARING (stash-then-clear pattern)
----------------------------------------------
Every chat/analysis tab (General, RAG, Vision, Data Analysis) clears its
message textbox via a two-step chained event instead of a single handler
that both answers the question AND clears the box:

    msg.submit(stash_fn, [msg], [msg, pending_state], queue=False).then(
        do_chat_fn, [pending_state, ...], [...]
    )

Why: the actual chat/agent call can take anywhere from a few seconds to
several minutes (agentic tabs with a slow local GGUF model hitting
max_steps can run for minutes — see general_agent.py/rag_agent.py module
docstrings). If that single combined call is interrupted by a dropped
connection, a reverse-proxy/browser timeout, or any server-side exception
that occurs outside the handler's own try/except, the textbox-clearing
output update never reaches the browser — even though chat.py's handlers
already return "" unconditionally in every normal case. The result is a
textbox that appears "stuck" with the old message after a failure,
especially on the longest-running tab (General Chat's agentic mode).

The fix: split "clear the input" into its own lightweight step that runs
FIRST, synchronously, with queue=False (so it fires immediately and isn't
stuck behind the slow call in the queue), and STASH the just-submitted
message into a gr.State first — never re-read the textbox's own value in
the second step, since by then the textbox has already been cleared
client-side and would hand the second function an empty string. This
guarantees the box empties the instant the message is sent, independent
of whether the subsequent chat/agent call succeeds, errors, hits
max_steps, or the connection drops entirely.
"""

import gradio as gr

import branding
import chat
import data_analysis
import deep_research_agent
import general_agent
import hardware
import knowledge_base as kb
import llama_backend
import model_registry as mr
import models
import rag_agent
from hardware import DEVICE
from i18n import LANGUAGES

APP_VERSION = branding.APP_VERSION

CSS = """
.status-bar   { font-size:0.82rem; color:#888; padding:4px 8px; }
.header-wrap  { display:flex; align-items:baseline; gap:12px; margin-bottom:6px; }
.header-title { font-size:1.6rem; font-weight:700; }
.header-sub   { font-size:0.9rem; color:#aaa; }
.dev-logo     { width:56px; height:56px; border-radius:50%; object-fit:cover;
                box-shadow:0 0 8px rgba(120,80,255,0.6); flex-shrink:0; margin-right:4px; }
.beta-badge   { display:inline-block; font-size:0.68rem; font-weight:700; letter-spacing:0.5px;
                color:#1a1a1a; background:#ffcc66; border-radius:999px; padding:2px 9px;
                margin-left:6px; vertical-align:middle; }
.tab-sidebar  { border-right:1px solid #444; padding-right:16px; margin-right:4px; }
.sidebar-hd   { margin-top:0 !important; opacity:0.85; }
.gpu-warning  { background:#3a2e0f; border:1px solid #a87c1f; border-radius:8px;
                padding:10px 14px; margin-bottom:10px; font-size:0.88rem; line-height:1.5; }
"""


def _gpu_warning_html(lang_key: str) -> str:
    """Build the (possibly empty) GPU-incompatibility warning banner HTML.

    Returns "" (nothing rendered) when the GPU is fine or there's no GPU
    at all — see hardware._detect_gpu_kernel_incompatibility(). Only
    non-empty when a CUDA GPU was detected but this PyTorch build has no
    compiled kernels for it (e.g. an old Pascal-class card like an MX230
    on a recent PyTorch release) — the exact case where the app silently
    falls back to CPU and a non-technical user would otherwise have no
    idea why, since torch.cuda.is_available() alone can't tell them.
    """
    info = hardware.get_gpu_incompatibility_info()
    if not info:
        return ""
    l = LANGUAGES.get(lang_key, LANGUAGES["kh"])
    msg = l["gpu_incompat_warning"].format(
        gpu=info["gpu_name"], cc=info["compute_capability"], archs=info["supported_archs"]
    )
    return f'<div class="gpu-warning">{msg}</div>'


def build_ui():
    # Start with Khmer as default
    L = LANGUAGES["kh"]

    with gr.Blocks(title=f"🤖 {branding.APP_NAME_EN} — LocalAiLab") as demo:
        lang_state = gr.State("kh")

        # ── Header ────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=8):
                with gr.Row():
                    if branding.DEVELOPER_LOGO_B64:
                        with gr.Column(scale=0, min_width=64):
                            gr.HTML(
                                f'<img src="data:image/jpeg;base64,{branding.DEVELOPER_LOGO_B64}" '
                                f'class="dev-logo" alt="{branding.DEVELOPER_NAME} logo" />'
                            )
                    with gr.Column():
                        header_title = gr.HTML(
                            f'<div class="header-wrap"><span class="header-title">{L["title"]}</span>'
                            f'<span class="beta-badge">BETA</span></div>'
                        )
                        header_sub = gr.HTML(
                            f'<div class="header-wrap"><span class="header-sub">'
                            f'{L["subtitle"].format(device=DEVICE.upper(), version=APP_VERSION)}</span></div>'
                        )
            with gr.Column(scale=2, variant="panel"):
                lang_dropdown = gr.Dropdown(
                    choices=["Khmer", "English"], value="Khmer",
                    label="🌐 Language", scale=1
                )

        # ── GPU-incompatibility warning banner (hidden when there's ──
        # nothing to warn about — see hardware.get_gpu_incompatibility_
        # info()). Built once at UI-build time from whatever DEVICE was
        # already resolved to at app-import time; updated on a language
        # switch via switch_lang() below so it re-renders in the newly
        # selected language instead of staying stuck in Khmer/English.
        gpu_warning_html = gr.HTML(value=_gpu_warning_html("kh"))

        # ── GGUF model folder (optional, user-configurable) ─────────
        # No path is hardcoded — leave blank to skip GGUF entirely, or
        # point it at any folder of .gguf files and click Scan. This can
        # also be pre-set via the LLAMA_CPP_MODEL_DIR environment variable.
        # The scan-result textbox stays hidden until a scan actually runs.
        with gr.Row():
            gguf_dir_tb = gr.Textbox(
                value=llama_backend.LLAMA_CPP_MODEL_DIR,
                placeholder=L["gguf_dir_placeholder"],
                label=L["label_gguf_dir"], scale=8,
            )
            scan_gguf_btn = gr.Button(L["btn_scan_gguf"], scale=1)
            ctx_window_dd = gr.Dropdown(
                choices=list(mr.CONTEXT_WINDOW_OPTIONS.keys()),
                value=mr.get_saved_context_window_label(),
                label=L["label_context_window"],
                info=L["info_context_window"],
                scale=3,
            )
        gguf_scan_status = gr.Textbox(show_label=False, interactive=False, visible=False)
        ctx_window_status = gr.Textbox(show_label=False, interactive=False, visible=False)
        with gr.Accordion(L["accordion_details"], open=False) as acc_ctx_detail:
            ctx_window_detail_md = gr.Markdown(L["info_context_window_detail"])

        # ── Tabs ──────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: General Chat ───────────────────────────────
            with gr.Tab(L["tab_general"]) as tab_gen:
                with gr.Row():
                    with gr.Column(scale=3, min_width=260, elem_classes=["tab-sidebar"]):
                        gen_settings_header = gr.Markdown(f"### {L['accordion_settings']}", elem_classes=["sidebar-hd"])
                        gen_desc = gr.Markdown(L["tab_general_desc"])
                        gen_agentic_chk = gr.Checkbox(
                            label=L["label_gen_agentic"], value=False,
                            info=L["info_gen_agentic"],
                        )
                        gen_memory_chk = gr.Checkbox(
                            label=L["label_memory"], value=True,
                            info=L["info_memory"],
                        )
                        with gr.Accordion(L["accordion_details"], open=False) as acc_gen_detail:
                            gen_agentic_detail_md = gr.Markdown(L["info_gen_agentic_detail"])
                            gen_memory_detail_md  = gr.Markdown(L["info_memory_detail"])
                        model_dd_gen = gr.Dropdown(choices=list(mr.MODEL_OPTIONS.keys()), value=mr.DEFAULT_LLM_LABEL, label=L["label_llm"])
                        with gr.Row():
                            reload_gen     = gr.Button(L["btn_load"], size="sm")
                            unload_gen_btn = gr.Button(L["btn_unload"], size="sm")
                        reload_gen_out = gr.Textbox(show_label=False, interactive=False, visible=False)
                        # Holds the just-submitted message across the
                        # stash -> clear -> answer chain (see module
                        # docstring: "INPUT-BOX CLEARING" above) — never
                        # re-read msg_gen's own value inside do_chat_general,
                        # since by then it has already been cleared.
                        pending_gen_msg = gr.State("")
                    with gr.Column(scale=7):
                        with gr.Accordion(L["accordion_chat"], open=False) as acc_gen_chat:
                            bot_gen   = gr.Chatbot(height=440)
                            clear_gen = gr.Button(L["btn_clear"], size="sm")
                        with gr.Row():
                            msg_gen  = gr.Textbox(placeholder=L["placeholder_gen"], show_label=False, scale=8)
                            send_gen = gr.Button(L["btn_send"], variant="primary", scale=1)

            # ── Tab 2: Vision Chat ────────────────────────────────
            with gr.Tab(L["tab_vision"]) as tab_vis:
                with gr.Row():
                    with gr.Column(scale=3, min_width=260, elem_classes=["tab-sidebar"]):
                        vis_settings_header = gr.Markdown(f"### {L['accordion_settings']}", elem_classes=["sidebar-hd"])
                        vis_desc = gr.Markdown(L["tab_vision_desc"])
                        vlm_dd      = gr.Dropdown(choices=list(mr.VLM_OPTIONS.keys()), value=mr.DEFAULT_VLM_LABEL, label=L["label_vlm"])
                        vis_rag_chk = gr.Checkbox(
                            label=L["label_vis_rag"], value=False,
                            info=L["label_vis_rag_info"],
                        )
                        vis_memory_chk = gr.Checkbox(
                            label=L["label_memory"], value=True,
                            info=L["info_memory"],
                        )
                        with gr.Accordion(L["accordion_details"], open=False) as acc_vis_detail:
                            vis_rag_detail_md    = gr.Markdown(L["label_vis_rag_info_detail"])
                            vis_memory_detail_md = gr.Markdown(L["info_memory_detail"])
                        with gr.Row():
                            load_vlm_btn   = gr.Button(L["btn_load"], size="sm")
                            unload_vlm_btn = gr.Button(L["btn_unload"], size="sm")
                        load_vlm_out = gr.Textbox(show_label=False, interactive=False, visible=False)
                        # See pending_gen_msg above — same stash-then-clear
                        # pattern applied to Vision Chat's message box.
                        pending_vis_msg = gr.State("")
                    with gr.Column(scale=7):
                        with gr.Accordion(L["accordion_chat"], open=False) as acc_vis_chat:
                            bot_vis   = gr.Chatbot(height=400)
                            clear_vis = gr.Button(L["btn_clear"], size="sm")
                        with gr.Row():
                            msg_vis    = gr.Textbox(placeholder=L["placeholder_vis"], show_label=False, scale=6)
                            img_upload = gr.Image(type="pil", sources=["upload", "clipboard"], scale=2)
                            send_vis   = gr.Button(L["btn_send"], variant="primary", scale=1)

            # ── Tab 3: Speech to Text ─────────────────────────────
            with gr.Tab(L["tab_stt"]) as tab_stt:
                with gr.Row():
                    with gr.Column(scale=3, min_width=260, elem_classes=["tab-sidebar"]):
                        stt_settings_header = gr.Markdown(f"### {L['accordion_settings']}", elem_classes=["sidebar-hd"])
                        stt_desc = gr.Markdown(L["tab_stt_desc"])
                        stt_dd      = gr.Dropdown(choices=list(mr.STT_OPTIONS.keys()), value=mr.DEFAULT_STT_LABEL, label=L["label_stt"])
                        stt_lang_dd = gr.Dropdown(
                            choices=[("Auto-detect", "auto"), ("English", "english"), ("Khmer", "khmer"),
                                     ("French", "french"), ("Chinese", "chinese"), ("Japanese", "japanese")],
                            value="auto", label=L["label_stt_lang"],
                        )
                        with gr.Row():
                            load_stt_btn   = gr.Button(L["btn_load"], size="sm")
                            unload_stt_btn = gr.Button(L["btn_unload"], size="sm")
                        load_stt_out = gr.Textbox(show_label=False, interactive=False, visible=False)
                        stt_hint = gr.Markdown(L["stt_khmer_hint"])
                        with gr.Accordion(L["accordion_details"], open=False) as acc_stt_detail:
                            stt_hint_detail_md = gr.Markdown(L["stt_khmer_hint_detail"])
                    with gr.Column(scale=7):
                        with gr.Accordion(f"📝 {L['label_res']}", open=False) as acc_stt_result:
                            stt_output = gr.Textbox(label=L["label_res"], lines=8, interactive=True)
                        stt_audio      = gr.Audio(label=L["stt_audio_label"], sources=["microphone", "upload"], type="filepath")
                        transcribe_btn = gr.Button(L["btn_transcribe"], variant="primary")

            # ── Tab 4: Data Analysis ──────────────────────────────
            with gr.Tab(L["tab_data"]) as tab_data:
                with gr.Row():
                    with gr.Column(scale=3, min_width=260, elem_classes=["tab-sidebar"]):
                        data_settings_header = gr.Markdown(f"### {L['accordion_settings']}", elem_classes=["sidebar-hd"])
                        data_desc = gr.Markdown(L["tab_data_desc"])
                        model_dd_data  = gr.Dropdown(choices=list(mr.MODEL_OPTIONS.keys()), value=mr.DEFAULT_LLM_LABEL, label=L["label_llm"])
                        data_memory_chk = gr.Checkbox(
                            label=L["label_memory"], value=True,
                            info=L["info_memory"],
                        )
                        with gr.Accordion(L["accordion_details"], open=False) as acc_data_detail:
                            data_memory_detail_md = gr.Markdown(L["info_memory_detail"])
                        reset_data_btn = gr.Button(L["btn_reset_agent"], size="sm")
                        reset_data_out = gr.Textbox(show_label=False, interactive=False, visible=False)
                        # See pending_gen_msg above — same stash-then-clear
                        # pattern applied to Data Analysis's question box.
                        pending_data_question = gr.State("")
                    with gr.Column(scale=7):
                        with gr.Accordion(L["accordion_chat"], open=False) as acc_data_chat:
                            bot_data   = gr.Chatbot(height=420)
                            clear_data = gr.Button(L["btn_clear"], size="sm")
                        with gr.Accordion(L["accordion_data_results"], open=False) as acc_data_results:
                            data_gallery     = gr.Gallery(label=L["label_charts"], columns=3, height=280)
                            data_report_file = gr.File(label=L["label_report_file"], interactive=False)
                        data_file_up = gr.File(label=L["data_file_label"], file_types=[".csv", ".xlsx", ".xls"], file_count="multiple")
                        with gr.Row():
                            msg_data  = gr.Textbox(placeholder=L["placeholder_data"], show_label=False, scale=8)
                            send_data = gr.Button(L["btn_send"], variant="primary", scale=1)

            # ── Tab 5: Knowledge Base ─────────────────────────────
            # No "⚙️ Settings" equivalent here — kept as a single column.
            with gr.Tab(L["tab_kb"]) as tab_kb:
                # Index stats (text chunks / visual index) — only relevant
                # here and on the RAG Chat tab (the two places retrieval
                # actually happens), not on every tab. Populated lazily via
                # demo.load() below (see that comment for why it isn't
                # computed inline at build time), and refreshed after any
                # upload/delete/clear action further down.
                kb_status_bar = gr.Textbox(
                    value="…", interactive=False,
                    show_label=False, elem_classes=["status-bar"]
                )
                with gr.Accordion(L["accordion_add"], open=True) as acc_add:
                    file_up    = gr.File(label=L["file_label"], file_types=[".pdf",".txt",".md",".docx"], file_count="multiple")
                    vis_ret_dd = gr.Dropdown(choices=list(mr.VISUAL_RETRIEVER_OPTIONS.keys()), value=list(mr.VISUAL_RETRIEVER_OPTIONS.keys())[0], label=L["label_vis_ret"])
                    with gr.Row():
                        up_btn             = gr.Button(L["btn_index"], variant="primary", scale=3)
                        unload_visual_btn  = gr.Button(L["btn_unload"], size="sm", scale=2)
                    up_msg = gr.Textbox(label=L["label_res"], interactive=False, lines=4, visible=False)

                with gr.Accordion(L["label_kb_docs"], open=False) as acc_kb_docs:
                    doc_table = gr.Dataframe(
                        headers=L["doc_table_headers"],
                        datatype=["str","str","str","number"],
                        value=kb.get_doc_table,
                        interactive=True, wrap=True,
                    )
                    with gr.Row():
                        refresh_btn    = gr.Button(L["btn_refresh"], size="sm", scale=2)
                        delete_sel_btn = gr.Button(L["btn_delete"], variant="stop", size="sm", scale=2)
                        clear_all_btn  = gr.Button(L["btn_clear_all"], variant="stop", size="sm", scale=2)
                    action_msg = gr.Textbox(label="", interactive=False, lines=1, visible=False)
                selected_rows_state = gr.State([])

            # ── Tab 7: RAG Chat (agentic — see rag_agent.py) ──────
            with gr.Tab(L["tab_rag"]) as tab_rag:
                with gr.Row():
                    with gr.Column(scale=3, min_width=260, elem_classes=["tab-sidebar"]):
                        rag_settings_header = gr.Markdown(f"### {L['accordion_settings']}", elem_classes=["sidebar-hd"])
                        rag_desc = gr.Markdown(L["tab_rag_desc"])
                        # Same index stats as the Knowledge Base tab (text
                        # chunks / visual index) — shown here too since RAG
                        # Chat is the other place retrieval actually matters.
                        rag_status_bar = gr.Textbox(
                            value="…", interactive=False,
                            show_label=False, elem_classes=["status-bar"]
                        )
                        rag_agentic_chk = gr.Checkbox(
                            label=L["label_rag_agentic"], value=True,
                            info=L["info_rag_agentic"],
                        )
                        rag_memory_chk = gr.Checkbox(
                            label=L["label_memory"], value=True,
                            info=L["info_memory"],
                        )
                        with gr.Accordion(L["accordion_details"], open=False) as acc_rag_detail:
                            rag_agentic_detail_md = gr.Markdown(L["info_rag_agentic_detail"])
                            rag_memory_detail_md  = gr.Markdown(L["info_memory_detail"])
                        model_dd_rag = gr.Dropdown(choices=list(mr.MODEL_OPTIONS.keys()), value=mr.DEFAULT_LLM_LABEL, label=L["label_llm"])
                        with gr.Row():
                            reload_rag     = gr.Button(L["btn_load"], size="sm")
                            unload_rag_btn = gr.Button(L["btn_unload"], size="sm")
                        reload_rag_out = gr.Textbox(show_label=False, interactive=False, visible=False)
                        # See pending_gen_msg above — same stash-then-clear
                        # pattern applied to RAG Chat's message box.
                        pending_rag_msg = gr.State("")
                    with gr.Column(scale=7):
                        with gr.Accordion(L["accordion_chat"], open=False) as acc_rag_chat:
                            bot_rag   = gr.Chatbot(height=440)
                            clear_rag = gr.Button(L["btn_clear"], size="sm")
                        with gr.Row():
                            msg_rag  = gr.Textbox(placeholder=L["placeholder_rag"], show_label=False, scale=8)
                            send_rag = gr.Button(L["btn_send"], variant="primary", scale=1)

            # ── Tab 8: Deep Research (manager + web-search sub-agent — ──
            # see deep_research_agent.py) ─────────────────────────────
            with gr.Tab(L["tab_deep_research"]) as tab_deep_research:
                with gr.Row():
                    with gr.Column(scale=3, min_width=260, elem_classes=["tab-sidebar"]):
                        dr_settings_header = gr.Markdown(f"### {L['accordion_settings']}", elem_classes=["sidebar-hd"])
                        dr_desc = gr.Markdown(L["tab_deep_research_desc"])
                        dr_memory_chk = gr.Checkbox(
                            label=L["label_memory"], value=True,
                            info=L["info_memory"],
                        )
                        with gr.Accordion(L["accordion_details"], open=False) as acc_dr_detail:
                            dr_memory_detail_md = gr.Markdown(L["info_memory_detail"])
                        model_dd_dr = gr.Dropdown(choices=list(mr.MODEL_OPTIONS.keys()), value=mr.DEFAULT_LLM_LABEL, label=L["label_llm"])
                        with gr.Row():
                            reload_dr     = gr.Button(L["btn_load"], size="sm")
                            unload_dr_btn = gr.Button(L["btn_unload"], size="sm")
                        reload_dr_out = gr.Textbox(show_label=False, interactive=False, visible=False)
                        # Rebuilds just the manager+search-agent wrapper
                        # (not the underlying LLM) — useful if a run gets
                        # stuck mid-delegation, same role as Data
                        # Analysis's reset button.
                        reset_dr_btn = gr.Button(L["btn_reset_agent"], size="sm")
                        reset_dr_out = gr.Textbox(show_label=False, interactive=False, visible=False)
                        # See pending_gen_msg above — same stash-then-clear
                        # pattern applied to Deep Research's message box.
                        pending_dr_msg = gr.State("")
                    with gr.Column(scale=7):
                        with gr.Accordion(L["accordion_chat"], open=False) as acc_dr_chat:
                            bot_dr   = gr.Chatbot(height=460)
                            clear_dr = gr.Button(L["btn_clear"], size="sm")
                        with gr.Row():
                            msg_dr  = gr.Textbox(placeholder=L["placeholder_deep_research"], show_label=False, scale=8)
                            send_dr = gr.Button(L["btn_send"], variant="primary", scale=1)

            # ── Tab 9: About ──────────────────────────────────────
            with gr.Tab(L["tab_about"]) as tab_about:
                about_md_kh = gr.Markdown(branding.about_content_kh(DEVICE.upper(), APP_VERSION))
                gr.Markdown("---")
                about_md_en = gr.Markdown(branding.about_content_en(DEVICE.upper(), APP_VERSION))

        # ── Event handlers ────────────────────────────────────────

        # GGUF model folder — rescan updates every model dropdown at once,
        # and reveals the scan-result textbox (hidden until a scan runs).
        def do_rescan_gguf(folder_path, lang_key):
            msg, dd1, dd2, dd3, vlm_dd_update = mr.rescan_gguf_models(folder_path, lang_key)
            return gr.update(value=msg, visible=True), dd1, dd2, dd3, vlm_dd_update

        scan_gguf_btn.click(
            do_rescan_gguf,
            [gguf_dir_tb, lang_state],
            [gguf_scan_status, model_dd_gen, model_dd_rag, model_dd_data, vlm_dd],
        )
        gguf_dir_tb.submit(
            do_rescan_gguf,
            [gguf_dir_tb, lang_state],
            [gguf_scan_status, model_dd_gen, model_dd_rag, model_dd_data, vlm_dd],
        )

        # Context Window (n_ctx) — GGUF/llama.cpp only. Persists the choice
        # immediately (models.get_llm()'s fallback logic picks it up on any
        # future load), and if a GGUF model happens to be loaded right now,
        # force-reloads it with the new n_ctx so the change is felt right
        # away instead of silently waiting for the next unrelated reload.
        def do_change_context_window(label):
            n_ctx = mr.CONTEXT_WINDOW_OPTIONS.get(label, mr.DEFAULT_CONTEXT_WINDOW)
            mr.set_context_window(n_ctx)
            # Every agentic CodeAgent wrapper holds its own reference to
            # the shared LLM object — drop all four caches so none of
            # them keep pointing at the model instance we're about to
            # release below (same pattern as reload_gen_fn/reload_rag_fn).
            general_agent.reset_agent()
            rag_agent.reset_agent()
            data_analysis.reset_agent()
            deep_research_agent.reset_agent()
            currently_loaded = models._llm_model_id
            if models._llm is not None and str(currently_loaded).lower().endswith(".gguf"):
                try:
                    models.force_reload_llm(currently_loaded, n_ctx=n_ctx)
                    msg = f"✅ Context window set to {n_ctx} tokens — '{currently_loaded}' reloaded."
                except Exception as e:
                    msg = f"❌ Failed to reload with the new context window: {e}"
            else:
                msg = f"✅ Context window set to {n_ctx} tokens — will apply next time a GGUF model loads."
            return gr.update(value=msg, visible=True)

        ctx_window_dd.change(do_change_context_window, [ctx_window_dd], [ctx_window_status])

        # General Chat
        def reload_gen_fn(label):
            # Both the general agentic CodeAgent and the data-analysis
            # agent hold their own reference to the shared LLM instance —
            # reset both caches so neither keeps the old model (or its
            # now-stale weights) alive; they rebuild cheaply against the
            # newly loaded one on next use.
            general_agent.reset_agent()
            data_analysis.reset_agent()
            mid = mr.MODEL_OPTIONS.get(label, mr.DEFAULT_LLM_MODEL)
            try:
                models.force_reload_llm(mid)
                return gr.update(value=f"✅ '{mid}' loaded", visible=True)
            except Exception as e:
                return gr.update(value=f"❌ {e}", visible=True)

        def unload_gen_fn(lang_key):
            msg = models.unload_llm_fn(lang_key)
            general_agent.reset_agent()
            data_analysis.reset_agent()
            return gr.update(value=msg, visible=True)

        def stash_gen(user_message):
            # Fires first, synchronously (queue=False) — clears msg_gen
            # immediately and hands the message off to pending_gen_msg for
            # the slow step below, instead of leaving it sitting in the
            # textbox for however long the (possibly multi-minute) agent
            # call takes. See module docstring: "INPUT-BOX CLEARING".
            return "", user_message

        def do_chat_general(pending_message, history, model_label, use_agentic, use_memory):
            # Wraps chat.chat_general() to also pop the conversation
            # accordion open the moment there's something to show — it
            # starts collapsed on every page load. Reads the message from
            # pending_gen_msg (stashed by stash_gen above), NOT from
            # msg_gen itself, which has already been cleared by the time
            # this runs.
            history, _ = chat.chat_general(pending_message, history, model_label, use_agentic, use_memory)
            return history, gr.update(open=True)

        msg_gen.submit(stash_gen, [msg_gen], [msg_gen, pending_gen_msg], queue=False).then(
            do_chat_general, [pending_gen_msg, bot_gen, model_dd_gen, gen_agentic_chk, gen_memory_chk],
            [bot_gen, acc_gen_chat]
        )
        send_gen.click(stash_gen, [msg_gen], [msg_gen, pending_gen_msg], queue=False).then(
            do_chat_general, [pending_gen_msg, bot_gen, model_dd_gen, gen_agentic_chk, gen_memory_chk],
            [bot_gen, acc_gen_chat]
        )

        def clear_gen_fn():
            # Clearing the visible chat should also clear the agentic
            # CodeAgent's own memory (agent.memory.steps) — otherwise a
            # "fresh" conversation would still secretly remember the old
            # one via agent.run(..., reset=False). Dropping the cached
            # agent object is enough: the next build starts with empty
            # memory (standard smolagents behaviour — see
            # agent_memory.py's module docstring for why this app no
            # longer persists memory to disk that would need clearing
            # separately here).
            general_agent.reset_agent()
            return [], "", gr.update(open=False)

        clear_gen.click(clear_gen_fn, outputs=[bot_gen, msg_gen, acc_gen_chat])
        reload_gen.click(reload_gen_fn, [model_dd_gen], [reload_gen_out])
        unload_gen_btn.click(unload_gen_fn, [lang_state], [reload_gen_out])

        # RAG Chat (agentic — see rag_agent.py — or direct, see chat.py)
        def reload_rag_fn(label):
            # The RAG CodeAgent and the data-analysis CodeAgent each hold
            # their own reference to the shared LLM instance — reset both
            # caches so neither keeps the old model (or stale weights)
            # alive; they rebuild cheaply against the newly loaded model.
            rag_agent.reset_agent()
            data_analysis.reset_agent()
            mid = mr.MODEL_OPTIONS.get(label, mr.DEFAULT_LLM_MODEL)
            try:
                models.force_reload_llm(mid)
                return gr.update(value=f"✅ '{mid}' loaded", visible=True)
            except Exception as e:
                return gr.update(value=f"❌ {e}", visible=True)

        def unload_rag_fn(lang_key):
            msg = models.unload_llm_fn(lang_key)
            rag_agent.reset_agent()
            data_analysis.reset_agent()
            return gr.update(value=msg, visible=True)

        def stash_rag(user_message):
            # See stash_gen() above.
            return "", user_message

        def do_chat_rag(pending_message, history, model_label, use_agentic, use_memory):
            history, _ = chat.chat_rag(pending_message, history, model_label, use_agentic, use_memory)
            return history, gr.update(open=True)

        msg_rag.submit(stash_rag, [msg_rag], [msg_rag, pending_rag_msg], queue=False).then(
            do_chat_rag, [pending_rag_msg, bot_rag, model_dd_rag, rag_agentic_chk, rag_memory_chk],
            [bot_rag, acc_rag_chat]
        )
        send_rag.click(stash_rag, [msg_rag], [msg_rag, pending_rag_msg], queue=False).then(
            do_chat_rag, [pending_rag_msg, bot_rag, model_dd_rag, rag_agentic_chk, rag_memory_chk],
            [bot_rag, acc_rag_chat]
        )

        def clear_rag_fn():
            # See clear_gen_fn() above — dropping the cached CodeAgent is
            # enough to guarantee a fresh, empty-memory agent next build.
            rag_agent.reset_agent()
            return [], "", gr.update(open=False)

        clear_rag.click(clear_rag_fn, outputs=[bot_rag, msg_rag, acc_rag_chat])
        reload_rag.click(reload_rag_fn, [model_dd_rag], [reload_rag_out])
        unload_rag_btn.click(unload_rag_fn, [lang_state], [reload_rag_out])

        # Deep Research (manager + web_search_agent — see deep_research_agent.py)
        def reload_dr_fn(label):
            deep_research_agent.reset_agent()
            mid = mr.MODEL_OPTIONS.get(label, mr.DEFAULT_LLM_MODEL)
            try:
                models.force_reload_llm(mid)
                return gr.update(value=f"✅ '{mid}' loaded", visible=True)
            except Exception as e:
                return gr.update(value=f"❌ {e}", visible=True)

        def unload_dr_fn(lang_key):
            msg = models.unload_llm_fn(lang_key)
            deep_research_agent.reset_agent()
            return gr.update(value=msg, visible=True)

        def reset_dr_agent_fn():
            deep_research_agent.reset_agent()
            return gr.update(value="✅ Agent reset — will rebuild on next run.", visible=True)

        def stash_dr(user_message):
            # See stash_gen() above.
            return "", user_message

        def do_chat_deep_research(pending_message, history, model_label, use_memory):
            history, _ = chat.chat_deep_research(pending_message, history, model_label, use_memory)
            return history, gr.update(open=True)

        msg_dr.submit(stash_dr, [msg_dr], [msg_dr, pending_dr_msg], queue=False).then(
            do_chat_deep_research, [pending_dr_msg, bot_dr, model_dd_dr, dr_memory_chk],
            [bot_dr, acc_dr_chat]
        )
        send_dr.click(stash_dr, [msg_dr], [msg_dr, pending_dr_msg], queue=False).then(
            do_chat_deep_research, [pending_dr_msg, bot_dr, model_dd_dr, dr_memory_chk],
            [bot_dr, acc_dr_chat]
        )

        def clear_dr_fn():
            # See clear_gen_fn() above — dropping the cached manager agent
            # (and its search sub-agent) is enough to guarantee a fresh,
            # empty-memory agent next build.
            deep_research_agent.reset_agent()
            return [], "", gr.update(open=False)

        clear_dr.click(clear_dr_fn, outputs=[bot_dr, msg_dr, acc_dr_chat])
        reload_dr.click(reload_dr_fn, [model_dd_dr], [reload_dr_out])
        unload_dr_btn.click(unload_dr_fn, [lang_state], [reload_dr_out])
        reset_dr_btn.click(reset_dr_agent_fn, outputs=[reset_dr_out])

        # Vision Chat
        def load_vlm_fn(label):
            mid = mr.VLM_OPTIONS.get(label, mr.DEFAULT_VLM_MODEL)
            try:
                models.force_reload_vlm(mid)
                return gr.update(value=f"✅ '{mid}' loaded", visible=True)
            except Exception as e:
                return gr.update(value=f"❌ {e}", visible=True)

        def unload_vlm_fn(lang_key):
            return gr.update(value=models.unload_vlm_fn(lang_key), visible=True)

        def stash_vis(user_message):
            # See stash_gen() above.
            return "", user_message

        def do_chat_vision(pending_message, uploaded_image, history, vlm_label, use_visual_rag, use_memory):
            history, img_reset = chat.chat_vision(pending_message, uploaded_image, history, vlm_label, use_visual_rag, use_memory)
            return history, img_reset, gr.update(open=True)

        send_vis.click(stash_vis, [msg_vis], [msg_vis, pending_vis_msg], queue=False).then(
            do_chat_vision, [pending_vis_msg, img_upload, bot_vis, vlm_dd, vis_rag_chk, vis_memory_chk],
            [bot_vis, img_upload, acc_vis_chat]
        )
        msg_vis.submit(stash_vis, [msg_vis], [msg_vis, pending_vis_msg], queue=False).then(
            do_chat_vision, [pending_vis_msg, img_upload, bot_vis, vlm_dd, vis_rag_chk, vis_memory_chk],
            [bot_vis, img_upload, acc_vis_chat]
        )
        clear_vis.click(lambda: ([], None, gr.update(open=False)), outputs=[bot_vis, img_upload, acc_vis_chat])
        load_vlm_btn.click(load_vlm_fn, [vlm_dd], [load_vlm_out])
        unload_vlm_btn.click(unload_vlm_fn, [lang_state], [load_vlm_out])

        # Speech to Text
        def load_stt_fn(label):
            mid = mr.STT_OPTIONS.get(label, mr.DEFAULT_STT_MODEL)
            try:
                models.force_reload_stt(mid)
                return gr.update(value=f"✅ '{mid}' loaded", visible=True)
            except Exception as e:
                return gr.update(value=f"❌ {e}", visible=True)

        def unload_stt_fn(lang_key):
            return gr.update(value=models.unload_stt_fn(lang_key), visible=True)

        def do_transcribe(audio_path, stt_label, lang_choice):
            mid = mr.STT_OPTIONS.get(stt_label, mr.DEFAULT_STT_MODEL)
            text = models.transcribe_audio(audio_path, language=lang_choice, model_id=mid)
            # Reveal (expand) the result accordion now that there's a
            # transcription to show — stays collapsed until this runs.
            return gr.update(value=text, visible=True), gr.update(open=True)

        transcribe_btn.click(do_transcribe, [stt_audio, stt_dd, stt_lang_dd], [stt_output, acc_stt_result])
        load_stt_btn.click(load_stt_fn, [stt_dd], [load_stt_out])
        unload_stt_btn.click(unload_stt_fn, [lang_state], [load_stt_out])

        # Data Analysis
        def reset_data_agent_fn():
            data_analysis.reset_agent()
            return gr.update(value="✅ Agent reset — will rebuild on next run.", visible=True)

        def stash_data(question):
            # See stash_gen() above.
            return "", question

        def do_data_analysis(files, pending_question, model_label, history, use_memory):
            history, gallery, report_file = data_analysis.run_data_analysis(files, pending_question, model_label, history, use_memory)
            # Reveal (expand) both the conversation and the results
            # accordions now that there's something in them — both stay
            # collapsed until an analysis actually runs.
            return history, gallery, report_file, gr.update(open=True), gr.update(open=True)

        send_data.click(stash_data, [msg_data], [msg_data, pending_data_question], queue=False).then(
            do_data_analysis, [data_file_up, pending_data_question, model_dd_data, bot_data, data_memory_chk],
            [bot_data, data_gallery, data_report_file, acc_data_chat, acc_data_results]
        )
        msg_data.submit(stash_data, [msg_data], [msg_data, pending_data_question], queue=False).then(
            do_data_analysis, [data_file_up, pending_data_question, model_dd_data, bot_data, data_memory_chk],
            [bot_data, data_gallery, data_report_file, acc_data_chat, acc_data_results]
        )

        def clear_data_fn():
            # See clear_gen_fn() above — dropping the cached agent is
            # enough to guarantee fresh memory next build. Also resets the
            # "last dataset" tracker so the next upload doesn't skip a
            # reset it should otherwise trigger.
            data_analysis.reset_agent()
            data_analysis._last_data_context["key"] = None
            return [], None, None, gr.update(open=False), gr.update(open=False)

        clear_data.click(clear_data_fn,
                         outputs=[bot_data, data_gallery, data_report_file, acc_data_chat, acc_data_results])
        reset_data_btn.click(reset_data_agent_fn, outputs=[reset_data_out])

        # Knowledge Base
        def on_select(evt: gr.SelectData, current):
            row = evt.index[0]
            if row in current: current.remove(row)
            else:              current.append(row)
            return current

        def do_upload(files, vis_ret_label, lang_key):
            import traceback
            try:
                msg = kb.index_uploaded_files(files, vis_ret_label)
            except Exception as e:
                msg = f"❌ {traceback.format_exc()}"
            # Reveal both the upload result and the (now updated) document
            # table — the table accordion stays collapsed until an upload,
            # refresh, delete, or clear actually happens. Also refresh the
            # index-stats textbox on both the Knowledge Base tab and the
            # RAG Chat tab, since an upload changes what both show.
            stats = kb.get_index_stats(lang_key)
            return gr.update(value=msg, visible=True), kb.get_doc_table(), gr.update(open=True), stats, stats

        def unload_visual_fn(lang_key):
            return gr.update(value=kb.unload_visual_retriever_fn(lang_key), visible=True)

        def do_refresh(lang_key):
            stats = kb.get_index_stats(lang_key)
            return kb.get_doc_table(), gr.update(open=True), stats, stats

        def do_delete(selected, table_data, lang_key):
            rows = table_data if isinstance(table_data, list) else table_data.values.tolist()
            new_table, msg = kb.delete_selected_sources(selected, rows)
            stats = kb.get_index_stats(lang_key)
            return new_table, gr.update(value=msg, visible=True), [], gr.update(open=True), stats, stats

        def do_clear(lang_key):
            table, msg = kb.clear_index()
            stats = kb.get_index_stats(lang_key)
            return table, gr.update(value=msg, visible=True), [], gr.update(open=True), stats, stats

        doc_table.select(on_select,        [selected_rows_state], [selected_rows_state])
        up_btn.click(do_upload,            [file_up, vis_ret_dd, lang_state], [up_msg, doc_table, acc_kb_docs, kb_status_bar, rag_status_bar])
        unload_visual_btn.click(unload_visual_fn, [lang_state], [up_msg])
        refresh_btn.click(do_refresh,      [lang_state], [doc_table, acc_kb_docs, kb_status_bar, rag_status_bar])
        delete_sel_btn.click(do_delete,    [selected_rows_state, doc_table, lang_state], [doc_table, action_msg, selected_rows_state, acc_kb_docs, kb_status_bar, rag_status_bar])
        clear_all_btn.click(do_clear,      [lang_state], [doc_table, action_msg, selected_rows_state, acc_kb_docs, kb_status_bar, rag_status_bar])

        # ── Language switcher ─────────────────────────────────────
        def switch_lang(lang_name):
            lk = "kh" if lang_name == "Khmer" else "en"
            l  = LANGUAGES[lk]
            return (
                lk,
                # header
                f'<div class="header-wrap"><span class="header-title">{l["title"]}</span>'
                f'<span class="beta-badge">BETA</span></div>',
                f'<div class="header-wrap"><span class="header-sub">{l["subtitle"].format(device=DEVICE.upper(), version=APP_VERSION)}</span></div>',
                # GPU-incompatibility warning banner, re-rendered in the
                # newly selected language (empty string if there's
                # nothing to warn about — see _gpu_warning_html()).
                _gpu_warning_html(lk),
                # GGUF model folder
                gr.update(label=l["label_gguf_dir"], placeholder=l["gguf_dir_placeholder"]),
                gr.update(value=l["btn_scan_gguf"]),
                gr.update(label=l["label_context_window"], info=l["info_context_window"]),
                # General Chat
                gr.update(placeholder=l["placeholder_gen"]),
                gr.update(value=l["btn_send"]),
                gr.update(value=l["btn_clear"]),
                gr.update(label=l["accordion_chat"]),
                gr.update(value=f"### {l['accordion_settings']}"),
                gr.update(value=l["tab_general_desc"]),
                gr.update(label=l["label_gen_agentic"], info=l["info_gen_agentic"]),
                gr.update(label=l["label_memory"], info=l["info_memory"]),
                gr.update(label=l["label_llm"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_unload"]),
                # RAG Chat
                gr.update(placeholder=l["placeholder_rag"]),
                gr.update(value=l["btn_send"]),
                gr.update(value=l["btn_clear"]),
                gr.update(label=l["accordion_chat"]),
                gr.update(value=f"### {l['accordion_settings']}"),
                gr.update(value=l["tab_rag_desc"]),
                gr.update(label=l["label_rag_agentic"], info=l["info_rag_agentic"]),
                gr.update(label=l["label_memory"], info=l["info_memory"]),
                gr.update(label=l["label_llm"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_unload"]),
                # Deep Research
                gr.update(placeholder=l["placeholder_deep_research"]),
                gr.update(value=l["btn_send"]),
                gr.update(value=l["btn_clear"]),
                gr.update(label=l["accordion_chat"]),
                gr.update(value=f"### {l['accordion_settings']}"),
                gr.update(value=l["tab_deep_research_desc"]),
                gr.update(label=l["label_memory"], info=l["info_memory"]),
                gr.update(label=l["label_llm"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_unload"]),
                gr.update(value=l["btn_reset_agent"]),
                # Vision Chat
                gr.update(placeholder=l["placeholder_vis"]),
                gr.update(value=l["btn_send"]),
                gr.update(value=l["btn_clear"]),
                gr.update(label=l["accordion_chat"]),
                gr.update(value=f"### {l['accordion_settings']}"),
                gr.update(value=l["tab_vision_desc"]),
                gr.update(label=l["label_vlm"]),
                gr.update(label=l["label_vis_rag"], info=l["label_vis_rag_info"]),
                gr.update(label=l["label_memory"], info=l["info_memory"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_unload"]),
                # STT
                gr.update(label=l["stt_audio_label"]),
                gr.update(value=l["btn_transcribe"]),
                gr.update(label=f"📝 {l['label_res']}"),
                gr.update(label=l["label_res"]),
                gr.update(value=f"### {l['accordion_settings']}"),
                gr.update(value=l["tab_stt_desc"]),
                gr.update(label=l["label_stt"]),
                gr.update(label=l["label_stt_lang"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_unload"]),
                gr.update(value=l["stt_khmer_hint"]),
                # Data Analysis
                gr.update(label=l["data_file_label"]),
                gr.update(placeholder=l["placeholder_data"]),
                gr.update(value=l["btn_send"]),
                gr.update(value=l["btn_clear"]),
                gr.update(label=l["accordion_chat"]),
                gr.update(label=l["accordion_data_results"]),
                gr.update(label=l["label_charts"]),
                gr.update(label=l["label_report_file"]),
                gr.update(value=f"### {l['accordion_settings']}"),
                gr.update(value=l["tab_data_desc"]),
                gr.update(label=l["label_llm"]),
                gr.update(label=l["label_memory"], info=l["info_memory"]),
                gr.update(value=l["btn_reset_agent"]),
                # Knowledge Base
                gr.update(label=l["accordion_add"]),
                gr.update(label=l["file_label"]),
                gr.update(label=l["label_vis_ret"]),
                gr.update(value=l["btn_index"]),
                gr.update(value=l["btn_unload"]),
                gr.update(label=l["label_res"]),
                gr.update(label=l["label_kb_docs"]),
                gr.update(value=l["btn_refresh"]),
                gr.update(value=l["btn_delete"]),
                gr.update(value=l["btn_clear_all"]),
                # Status bars (Knowledge Base tab + RAG Chat tab)
                kb.get_index_stats(lk),
                kb.get_index_stats(lk),
                # "Details" accordions (collapsed long explanations) + their content
                gr.update(label=l["accordion_details"]), gr.update(value=l["info_context_window_detail"]),
                gr.update(label=l["accordion_details"]), gr.update(value=l["info_gen_agentic_detail"]), gr.update(value=l["info_memory_detail"]),
                gr.update(label=l["accordion_details"]), gr.update(value=l["info_rag_agentic_detail"]), gr.update(value=l["info_memory_detail"]),
                gr.update(label=l["accordion_details"]), gr.update(value=l["info_memory_detail"]),
                gr.update(label=l["accordion_details"]), gr.update(value=l["label_vis_rag_info_detail"]), gr.update(value=l["info_memory_detail"]),
                gr.update(label=l["accordion_details"]), gr.update(value=l["stt_khmer_hint_detail"]),
                gr.update(label=l["accordion_details"]), gr.update(value=l["info_memory_detail"]),
            )

        _lang_outputs = [
            lang_state, header_title, header_sub, gpu_warning_html,
            gguf_dir_tb, scan_gguf_btn, ctx_window_dd,
            # General Chat
            msg_gen, send_gen, clear_gen, acc_gen_chat, gen_settings_header, gen_desc, gen_agentic_chk, gen_memory_chk, model_dd_gen, reload_gen, unload_gen_btn,
            # RAG Chat
            msg_rag, send_rag, clear_rag, acc_rag_chat, rag_settings_header, rag_desc, rag_agentic_chk, rag_memory_chk, model_dd_rag, reload_rag, unload_rag_btn,
            # Deep Research
            msg_dr, send_dr, clear_dr, acc_dr_chat, dr_settings_header, dr_desc, dr_memory_chk, model_dd_dr, reload_dr, unload_dr_btn, reset_dr_btn,
            # Vision Chat
            msg_vis, send_vis, clear_vis, acc_vis_chat, vis_settings_header, vis_desc, vlm_dd, vis_rag_chk, vis_memory_chk, load_vlm_btn, unload_vlm_btn,
            # STT
            stt_audio, transcribe_btn, acc_stt_result, stt_output, stt_settings_header, stt_desc,
            stt_dd, stt_lang_dd, load_stt_btn, unload_stt_btn, stt_hint,
            # Data Analysis
            data_file_up, msg_data, send_data, clear_data, acc_data_chat, acc_data_results, data_gallery, data_report_file,
            data_settings_header, data_desc, model_dd_data, data_memory_chk, reset_data_btn,
            # Knowledge Base
            acc_add, file_up, vis_ret_dd, up_btn, unload_visual_btn, up_msg,
            acc_kb_docs, refresh_btn, delete_sel_btn, clear_all_btn,
            # Status bars (Knowledge Base tab + RAG Chat tab)
            kb_status_bar, rag_status_bar,
            # "Details" accordions (collapsed long explanations)
            acc_ctx_detail, ctx_window_detail_md,
            acc_gen_detail, gen_agentic_detail_md, gen_memory_detail_md,
            acc_rag_detail, rag_agentic_detail_md, rag_memory_detail_md,
            acc_dr_detail, dr_memory_detail_md,
            acc_vis_detail, vis_rag_detail_md, vis_memory_detail_md,
            acc_stt_detail, stt_hint_detail_md,
            acc_data_detail, data_memory_detail_md,
        ]

        lang_dropdown.change(switch_lang, [lang_dropdown], _lang_outputs)

        # NOTE: no demo.load() re-initializer here. Every component above
        # is already created with the correct Khmer text/value via `L`
        # (e.g. label=L["label_llm"], placeholder=L["placeholder_gen"]),
        # so a startup call re-pushing the same Khmer values into all 68
        # components across every tab is redundant — and firing that many
        # updates into tabs the browser hasn't finished mounting yet (any
        # tab other than the default-active one) can leave those
        # components stuck showing a perpetual loading state. Only
        # lang_dropdown.change() needs to touch all of them, and that only
        # runs after the user explicitly switches languages, well after
        # the initial page mount has finished.

        # Narrow demo.load(): populates ONLY the two index-stats textboxes
        # (Knowledge Base tab + RAG Chat tab), not the 68-component
        # language re-init the comment above avoids. This is what actually
        # defers the ChromaDB client open until after the UI has mounted,
        # instead of during build_ui().
        demo.load(kb.get_index_stats, [lang_state], [kb_status_bar])
        demo.load(kb.get_index_stats, [lang_state], [rag_status_bar])

        return demo
