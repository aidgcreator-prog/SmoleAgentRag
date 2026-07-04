"""
ui.py — Builds the Gradio Blocks UI and wires every event handler.

All handlers are defined and wired at the top level of build_ui() (never
inside a gr.render() block) — see the "Gradio 6 breaks event handler
rebinding on render cycles" learning: defining .click()/.submit() inside a
@gr.render() causes all buttons/dropdowns to silently stop working after
the first language switch.
"""

import gradio as gr

import branding
import chat
import data_analysis
import knowledge_base as kb
import llama_backend
import model_registry as mr
import models
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
"""


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

        # ── GGUF model folder (optional, user-configurable) ─────────
        # No path is hardcoded — leave blank to skip GGUF entirely, or
        # point it at any folder of .gguf files and click Scan. This can
        # also be pre-set via the LLAMA_CPP_MODEL_DIR environment variable.
        with gr.Row():
            gguf_dir_tb = gr.Textbox(
                value=llama_backend.LLAMA_CPP_MODEL_DIR,
                placeholder=L["gguf_dir_placeholder"],
                label=L["label_gguf_dir"], scale=8,
            )
            scan_gguf_btn = gr.Button(L["btn_scan_gguf"], scale=1)
        gguf_scan_status = gr.Textbox(show_label=False, interactive=False)

        # ── Tabs ──────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: General Chat ───────────────────────────────
            with gr.Tab(L["tab_general"]) as tab_gen:
                gen_desc    = gr.Markdown(L["tab_general_desc"])
                bot_gen     = gr.Chatbot(height=440)
                with gr.Row():
                    msg_gen  = gr.Textbox(placeholder=L["placeholder_gen"], show_label=False, scale=8)
                    send_gen = gr.Button(L["btn_send"], variant="primary", scale=1)
                with gr.Row():
                    model_dd_gen   = gr.Dropdown(choices=list(mr.MODEL_OPTIONS.keys()), value=mr.DEFAULT_LLM_LABEL, label=L["label_llm"], scale=6)
                    reload_gen     = gr.Button(L["btn_load"], size="sm", scale=2)
                    unload_gen_btn = gr.Button(L["btn_unload"], size="sm", scale=2)
                    reload_gen_out = gr.Textbox(show_label=False, interactive=False, scale=4)
                clear_gen = gr.Button(L["btn_clear"], size="sm")

            # ── Tab 2: RAG Chat ───────────────────────────────────
            with gr.Tab(L["tab_rag"]) as tab_rag:
                rag_desc    = gr.Markdown(L["tab_rag_desc"])
                bot_rag     = gr.Chatbot(height=440)
                with gr.Row():
                    msg_rag  = gr.Textbox(placeholder=L["placeholder_rag"], show_label=False, scale=8)
                    send_rag = gr.Button(L["btn_send"], variant="primary", scale=1)
                with gr.Row():
                    model_dd_rag   = gr.Dropdown(choices=list(mr.MODEL_OPTIONS.keys()), value=mr.DEFAULT_LLM_LABEL, label=L["label_llm"], scale=6)
                    reload_rag     = gr.Button(L["btn_load"], size="sm", scale=2)
                    unload_rag_btn = gr.Button(L["btn_unload"], size="sm", scale=2)
                    reload_rag_out = gr.Textbox(show_label=False, interactive=False, scale=4)
                clear_rag = gr.Button(L["btn_clear"], size="sm")

            # ── Tab 3: Vision Chat ────────────────────────────────
            with gr.Tab(L["tab_vision"]) as tab_vis:
                vis_desc    = gr.Markdown(L["tab_vision_desc"])
                bot_vis     = gr.Chatbot(height=400)
                with gr.Row():
                    msg_vis    = gr.Textbox(placeholder=L["placeholder_vis"], show_label=False, scale=6)
                    img_upload = gr.Image(type="pil", sources=["upload", "clipboard"], scale=2)
                    send_vis   = gr.Button(L["btn_send"], variant="primary", scale=1)
                with gr.Row():
                    vlm_dd       = gr.Dropdown(choices=list(mr.VLM_OPTIONS.keys()), value=mr.DEFAULT_VLM_LABEL, label=L["label_vlm"], scale=5)
                    vis_rag_chk  = gr.Checkbox(
                        label=L["label_vis_rag"], value=False, scale=2,
                        info=L["label_vis_rag_info"],
                    )
                    load_vlm_btn = gr.Button(L["btn_load"], size="sm", scale=2)
                    unload_vlm_btn = gr.Button(L["btn_unload"], size="sm", scale=2)
                    load_vlm_out = gr.Textbox(show_label=False, interactive=False, scale=3)
                clear_vis = gr.Button(L["btn_clear"], size="sm")

            # ── Tab 4: Speech to Text ─────────────────────────────
            with gr.Tab(L["tab_stt"]) as tab_stt:
                stt_desc  = gr.Markdown(L["tab_stt_desc"])
                stt_audio = gr.Audio(label=L["stt_audio_label"], sources=["microphone", "upload"], type="filepath")
                with gr.Row():
                    stt_dd      = gr.Dropdown(choices=list(mr.STT_OPTIONS.keys()), value=mr.DEFAULT_STT_LABEL, label=L["label_stt"], scale=5)
                    stt_lang_dd = gr.Dropdown(
                        choices=[("Auto-detect", "auto"), ("English", "english"), ("Khmer", "khmer"),
                                 ("French", "french"), ("Chinese", "chinese"), ("Japanese", "japanese")],
                        value="auto", label=L["label_stt_lang"], scale=3,
                    )
                    load_stt_btn = gr.Button(L["btn_load"], size="sm", scale=2)
                    unload_stt_btn = gr.Button(L["btn_unload"], size="sm", scale=2)
                    load_stt_out = gr.Textbox(show_label=False, interactive=False, scale=3)
                stt_hint = gr.Markdown(L["stt_khmer_hint"])
                transcribe_btn = gr.Button(L["btn_transcribe"], variant="primary")
                stt_output     = gr.Textbox(label=L["label_res"], lines=8, interactive=True)

            # ── Tab 5: Data Analysis ──────────────────────────────
            with gr.Tab(L["tab_data"]) as tab_data:
                data_desc    = gr.Markdown(L["tab_data_desc"])
                data_file_up = gr.File(label=L["data_file_label"], file_types=[".csv", ".xlsx", ".xls"], file_count="multiple")
                bot_data     = gr.Chatbot(height=420)
                with gr.Row():
                    msg_data  = gr.Textbox(placeholder=L["placeholder_data"], show_label=False, scale=8)
                    send_data = gr.Button(L["btn_send"], variant="primary", scale=1)
                with gr.Row():
                    model_dd_data  = gr.Dropdown(choices=list(mr.MODEL_OPTIONS.keys()), value=mr.DEFAULT_LLM_LABEL, label=L["label_llm"], scale=6)
                    reset_data_btn = gr.Button(L["btn_reset_agent"], size="sm", scale=2)
                    reset_data_out = gr.Textbox(show_label=False, interactive=False, scale=4)
                data_gallery     = gr.Gallery(label=L["label_charts"], columns=3, height=280)
                data_report_file = gr.File(label=L["label_report_file"], interactive=False)
                clear_data = gr.Button(L["btn_clear"], size="sm")

            # ── Tab 6: Knowledge Base ─────────────────────────────
            with gr.Tab(L["tab_kb"]) as tab_kb:
                with gr.Accordion(L["accordion_add"], open=True) as acc_add:
                    file_up    = gr.File(label=L["file_label"], file_types=[".pdf",".txt",".md",".docx"], file_count="multiple")
                    vis_ret_dd = gr.Dropdown(choices=list(mr.VISUAL_RETRIEVER_OPTIONS.keys()), value=list(mr.VISUAL_RETRIEVER_OPTIONS.keys())[0], label=L["label_vis_ret"])
                    with gr.Row():
                        up_btn             = gr.Button(L["btn_index"], variant="primary", scale=3)
                        unload_visual_btn  = gr.Button(L["btn_unload"], size="sm", scale=2)
                    up_msg     = gr.Textbox(label=L["label_res"], interactive=False, lines=4)

                kb_header = gr.Markdown(L["header_docs"])
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
                action_msg         = gr.Textbox(label="", interactive=False, lines=1)
                selected_rows_state = gr.State([])

            # ── Tab 7: About ──────────────────────────────────────
            with gr.Tab(L["tab_about"]) as tab_about:
                about_md_kh = gr.Markdown(branding.about_content_kh(DEVICE.upper(), APP_VERSION))
                gr.Markdown("---")
                about_md_en = gr.Markdown(branding.about_content_en(DEVICE.upper(), APP_VERSION))

        status_bar = gr.Textbox(
            value=kb.get_index_stats("kh"), interactive=False,
            show_label=False, elem_classes=["status-bar"]
        )

        # ── Event handlers ────────────────────────────────────────

        # GGUF model folder — rescan updates every model dropdown at once
        scan_gguf_btn.click(
            mr.rescan_gguf_models,
            [gguf_dir_tb, lang_state],
            [gguf_scan_status, model_dd_gen, model_dd_rag, model_dd_data],
        )
        gguf_dir_tb.submit(
            mr.rescan_gguf_models,
            [gguf_dir_tb, lang_state],
            [gguf_scan_status, model_dd_gen, model_dd_rag, model_dd_data],
        )

        # General Chat
        def reload_gen_fn(label):
            # The data-analysis agent holds its own reference to the shared
            # LLM instance — reset it too so it doesn't keep the old model
            # (or its now-stale weights) alive; it rebuilds cheaply against
            # the newly loaded one on next use.
            data_analysis.reset_agent()
            mid = mr.MODEL_OPTIONS.get(label, mr.DEFAULT_LLM_MODEL)
            try:
                models.force_reload_llm(mid)
                return f"✅ '{mid}' loaded"
            except Exception as e:
                return f"❌ {e}"

        def unload_gen_fn(lang_key):
            msg = models.unload_llm_fn(lang_key)
            data_analysis.reset_agent()
            return msg

        msg_gen.submit(chat.chat_general,  [msg_gen, bot_gen, model_dd_gen], [bot_gen, msg_gen])
        send_gen.click(chat.chat_general,  [msg_gen, bot_gen, model_dd_gen], [bot_gen, msg_gen])
        clear_gen.click(lambda: ([], ""), outputs=[bot_gen, msg_gen])
        reload_gen.click(reload_gen_fn, [model_dd_gen], [reload_gen_out])
        unload_gen_btn.click(unload_gen_fn, [lang_state], [reload_gen_out])

        # RAG Chat
        def reload_rag_fn(label):
            data_analysis.reset_agent()
            mid = mr.MODEL_OPTIONS.get(label, mr.DEFAULT_LLM_MODEL)
            try:
                models.force_reload_llm(mid)
                return f"✅ '{mid}' loaded"
            except Exception as e:
                return f"❌ {e}"

        msg_rag.submit(chat.chat_rag,  [msg_rag, bot_rag, model_dd_rag], [bot_rag, msg_rag])
        send_rag.click(chat.chat_rag,  [msg_rag, bot_rag, model_dd_rag], [bot_rag, msg_rag])
        clear_rag.click(lambda: ([], ""), outputs=[bot_rag, msg_rag])
        reload_rag.click(reload_rag_fn, [model_dd_rag], [reload_rag_out])
        unload_rag_btn.click(unload_gen_fn, [lang_state], [reload_rag_out])

        # Vision Chat
        def load_vlm_fn(label):
            mid = mr.VLM_OPTIONS.get(label, mr.DEFAULT_VLM_MODEL)
            try:
                models.force_reload_vlm(mid)
                return f"✅ '{mid}' loaded"
            except Exception as e:
                return f"❌ {e}"

        send_vis.click(chat.chat_vision,  [msg_vis, img_upload, bot_vis, vlm_dd, vis_rag_chk], [bot_vis, img_upload])
        msg_vis.submit(chat.chat_vision,  [msg_vis, img_upload, bot_vis, vlm_dd, vis_rag_chk], [bot_vis, img_upload])
        clear_vis.click(lambda: ([], None), outputs=[bot_vis, img_upload])
        load_vlm_btn.click(load_vlm_fn, [vlm_dd], [load_vlm_out])
        unload_vlm_btn.click(models.unload_vlm_fn, [lang_state], [load_vlm_out])

        # Speech to Text
        def load_stt_fn(label):
            mid = mr.STT_OPTIONS.get(label, mr.DEFAULT_STT_MODEL)
            try:
                models.force_reload_stt(mid)
                return f"✅ '{mid}' loaded"
            except Exception as e:
                return f"❌ {e}"

        def do_transcribe(audio_path, stt_label, lang_choice):
            mid = mr.STT_OPTIONS.get(stt_label, mr.DEFAULT_STT_MODEL)
            return models.transcribe_audio(audio_path, language=lang_choice, model_id=mid)

        transcribe_btn.click(do_transcribe, [stt_audio, stt_dd, stt_lang_dd], [stt_output])
        load_stt_btn.click(load_stt_fn, [stt_dd], [load_stt_out])
        unload_stt_btn.click(models.unload_stt_fn, [lang_state], [load_stt_out])

        # Data Analysis
        def reset_data_agent_fn():
            data_analysis.reset_agent()
            return "✅ Agent reset — will rebuild on next run."

        def do_data_analysis(files, question, model_label, history):
            history, gallery, report_file = data_analysis.run_data_analysis(files, question, model_label, history)
            return history, "", gallery, report_file

        send_data.click(do_data_analysis, [data_file_up, msg_data, model_dd_data, bot_data],
                        [bot_data, msg_data, data_gallery, data_report_file])
        msg_data.submit(do_data_analysis, [data_file_up, msg_data, model_dd_data, bot_data],
                        [bot_data, msg_data, data_gallery, data_report_file])
        clear_data.click(lambda: ([], None, None), outputs=[bot_data, data_gallery, data_report_file])
        reset_data_btn.click(reset_data_agent_fn, outputs=[reset_data_out])

        # Knowledge Base
        def on_select(evt: gr.SelectData, current):
            row = evt.index[0]
            if row in current: current.remove(row)
            else:              current.append(row)
            return current

        def do_upload(files, vis_ret_label):
            import traceback
            try:
                return kb.index_uploaded_files(files, vis_ret_label), kb.get_doc_table()
            except Exception as e:
                return f"❌ {traceback.format_exc()}", kb.get_doc_table()

        def do_delete(selected, table_data):
            rows = table_data if isinstance(table_data, list) else table_data.values.tolist()
            new_table, msg = kb.delete_selected_sources(selected, rows)
            return new_table, msg, []

        def do_clear():
            table, msg = kb.clear_index()
            return table, msg, []

        doc_table.select(on_select,        [selected_rows_state], [selected_rows_state])
        up_btn.click(do_upload,            [file_up, vis_ret_dd], [up_msg, doc_table])
        unload_visual_btn.click(kb.unload_visual_retriever_fn, [lang_state], [up_msg])
        refresh_btn.click(kb.get_doc_table,   outputs=[doc_table])
        delete_sel_btn.click(do_delete,    [selected_rows_state, doc_table], [doc_table, action_msg, selected_rows_state])
        clear_all_btn.click(do_clear,      outputs=[doc_table, action_msg, selected_rows_state])

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
                # GGUF model folder
                gr.update(label=l["label_gguf_dir"], placeholder=l["gguf_dir_placeholder"]),
                gr.update(value=l["btn_scan_gguf"]),
                # General Chat
                gr.update(value=l["tab_general_desc"]),
                gr.update(placeholder=l["placeholder_gen"]),
                gr.update(value=l["btn_send"]),
                gr.update(label=l["label_llm"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_unload"]),
                gr.update(value=l["btn_clear"]),
                # RAG Chat
                gr.update(value=l["tab_rag_desc"]),
                gr.update(placeholder=l["placeholder_rag"]),
                gr.update(value=l["btn_send"]),
                gr.update(label=l["label_llm"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_unload"]),
                gr.update(value=l["btn_clear"]),
                # Vision Chat
                gr.update(value=l["tab_vision_desc"]),
                gr.update(placeholder=l["placeholder_vis"]),
                gr.update(value=l["btn_send"]),
                gr.update(label=l["label_vlm"]),
                gr.update(label=l["label_vis_rag"], info=l["label_vis_rag_info"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_unload"]),
                gr.update(value=l["btn_clear"]),
                # STT
                gr.update(value=l["tab_stt_desc"]),
                gr.update(label=l["stt_audio_label"]),
                gr.update(label=l["label_stt"]),
                gr.update(label=l["label_stt_lang"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_unload"]),
                gr.update(value=l["stt_khmer_hint"]),
                gr.update(value=l["btn_transcribe"]),
                gr.update(label=l["label_res"]),
                # Data Analysis
                gr.update(value=l["tab_data_desc"]),
                gr.update(label=l["data_file_label"]),
                gr.update(placeholder=l["placeholder_data"]),
                gr.update(value=l["btn_send"]),
                gr.update(label=l["label_llm"]),
                gr.update(value=l["btn_reset_agent"]),
                gr.update(label=l["label_charts"]),
                gr.update(label=l["label_report_file"]),
                gr.update(value=l["btn_clear"]),
                # Knowledge Base
                gr.update(label=l["file_label"]),
                gr.update(label=l["label_vis_ret"]),
                gr.update(value=l["btn_index"]),
                gr.update(value=l["btn_unload"]),
                gr.update(label=l["label_res"]),
                gr.update(value=l["header_docs"]),
                gr.update(value=l["btn_refresh"]),
                gr.update(value=l["btn_delete"]),
                gr.update(value=l["btn_clear_all"]),
                # Status bar
                kb.get_index_stats(lk),
            )

        _lang_outputs = [
            lang_state, header_title, header_sub,
            gguf_dir_tb, scan_gguf_btn,
            # General Chat
            gen_desc, msg_gen, send_gen, model_dd_gen, reload_gen, unload_gen_btn, clear_gen,
            # RAG Chat
            rag_desc, msg_rag, send_rag, model_dd_rag, reload_rag, unload_rag_btn, clear_rag,
            # Vision Chat
            vis_desc, msg_vis, send_vis, vlm_dd, vis_rag_chk, load_vlm_btn, unload_vlm_btn, clear_vis,
            # STT
            stt_desc, stt_audio, stt_dd, stt_lang_dd, load_stt_btn, unload_stt_btn, stt_hint, transcribe_btn, stt_output,
            # Data Analysis
            data_desc, data_file_up, msg_data, send_data, model_dd_data, reset_data_btn,
            data_gallery, data_report_file, clear_data,
            # Knowledge Base
            file_up, vis_ret_dd, up_btn, unload_visual_btn, up_msg,
            kb_header, refresh_btn, delete_sel_btn, clear_all_btn,
            # Status bar
            status_bar,
        ]

        lang_dropdown.change(switch_lang, [lang_dropdown], _lang_outputs)

        # Initialise to Khmer on load
        demo.load(lambda: switch_lang("Khmer"), outputs=_lang_outputs)

        return demo
