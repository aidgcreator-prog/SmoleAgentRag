# អាជ្ញាប័ណ្ណ និងសេចក្តីជូនដំណឹងភាគីទីបី / Third-Party Licenses & Notices

🇰🇭 **[ខ្មែរ](#-ខ្មែរ)** | 🇬🇧 **[English](#-english)**

---

## 🇰🇭 ខ្មែរ

កូដប្រភពរបស់គម្រោងនេះផ្ទាល់ (`app.py`, `index_docs.py` និងស្គ្រីបពាក់ព័ន្ធ)
ត្រូវបានផ្តល់អាជ្ញាប័ណ្ណក្រោម **Apache License 2.0** — សូមមើលឯកសារ
[`LICENSE`](./LICENSE) ដែលដូចគ្នានឹងអាជ្ញាប័ណ្ណរបស់
[`smolagents`](https://github.com/huggingface/smolagents) ជាក្របខណ្ឌ
ភ្នាក់ងារដែលកម្មវិធីនេះបានសាងសង់ឡើង។

អាជ្ញាប័ណ្ណ Apache-2.0 គ្របដណ្តប់ *តែកូដក្នុងឃ្លាំងនេះប៉ុណ្ណោះ*។ វា
**មិន**ផ្តល់អាជ្ញាប័ណ្ណឡើងវិញលើបណ្ណាល័យភាគីទីបី និងម៉ូដែល AI ដែលកម្មវិធី
ទាញយក និងដំណើរការនៅពេលដំណើរការនោះទេ — នីមួយៗនៅតែរក្សាអាជ្ញាប័ណ្ណផ្ទាល់ខ្លួន
ដែលអ្នកត្រូវគោរពដោយឡែកពីគ្នា។ សេចក្តីសង្ខេបមាននៅខាងក្រោម។ **នេះមិនមែនជា
ការណែនាំផ្នែកច្បាប់ទេ** — សូមពិនិត្យប្រភពដើមសម្រាប់ធាតុនីមួយៗ ប្រសិនបើអ្នក
មានគម្រោងចែកចាយឡើងវិញ ឬប្រើប្រាស់ក្នុងគោលបំណងពាណិជ្ជកម្ម។

---

### ⚠️ សំខាន់៖ PyMuPDF ប្រើអាជ្ញាប័ណ្ណ AGPL-3.0 មិនមែនប្រភេទសេរីទេ

`requirements.txt` រួមបញ្ចូល `PyMuPDF` (នាំចូលជា `fitz` នៅក្នុង `app.py`
ប្រើសម្រាប់ការដកស្រង់អត្ថបទ PDF នៅក្នុង `index_pdf_file()`)។ ខុសពី
ធាតុផ្សេងទៀតក្នុងស្តាកនេះ PyMuPDF/MuPDF ត្រូវបានផ្តល់អាជ្ញាប័ណ្ណទ្វេដោយ
Artifex Software ក្រោម៖

- **GNU AGPL v3** (ឥតគិតថ្លៃ ប៉ុន្តែជាប្រភេទ copyleft — ប្រសិនបើអ្នកចែកចាយ
  កម្មវិធីនេះ ឬដំណើរការវាជាសេវាបណ្តាញដែលអ្នកដទៃប្រើប្រាស់ កាតព្វកិច្ច
  ការបង្ហាញកូដប្រភពរបស់ AGPL អាចអនុវត្តចំពោះកម្មវិធីទាំងមូលរបស់អ្នក) ឬ
- **អាជ្ញាប័ណ្ណពាណិជ្ជកម្មដែលបង់ថ្លៃពី Artifex** (ដកចេញនូវកាតព្វកិច្ច AGPL)។

ប្រសិនបើអ្នកមានគម្រោងរក្សាគម្រោងនេះជា closed-source ចែកចាយវាជាពាណិជ្ជកម្ម
ឬផ្តល់ជូនជាសេវាដែលបង្ហោះ សូមទទួលបានអាជ្ញាប័ណ្ណពាណិជ្ជកម្ម MuPDF ពី Artifex
ឬប្តូរ `PyMuPDF` ទៅជាជម្រើសអាជ្ញាប័ណ្ណសេរីជំនួសវិញ (ឧ. `pdfplumber`
ដែលមានអាជ្ញាប័ណ្ណ MIT ទោះបីជាយឺតជាងក៏ដោយ)។ ប្រសិនបើគម្រោងនេះនៅតែជាឧបករណ៍
ផ្ទាល់ខ្លួន/មូលដ្ឋានដែលអ្នកដំណើរការតែម្នាក់ឯង កាតព្វកិច្ចរបស់ AGPL ទំនងជា
មិនប៉ះពាល់ខ្លាំងទេ ប៉ុន្តែអ្នកគួរតែដឹងអំពីវានៅតែដដែល។

---

### ម៉ូដែល AI

ម៉ូដែលត្រូវបានទាញយកនៅពេលដំណើរការពី Hugging Face ហើយ **មិន**ត្រូវបានចែកចាយ
ឡើងវិញជាផ្នែកមួយនៃឃ្លាំងនេះទេ។ នីមួយៗរក្សានូវលក្ខខណ្ឌអាជ្ញាប័ណ្ណផ្ទាល់ខ្លួន
ការប្រើប្រាស់ម៉ូដែលមានន័យថាយល់ព្រមតាមលក្ខខណ្ឌទាំងនោះដោយផ្ទាល់ជាមួយម្ចាស់
ម៉ូដែល។

| ម៉ូដែល | ប្រើសម្រាប់ | អាជ្ញាប័ណ្ណ |
|---|---|---|
| `Qwen/Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B` | Text LLM | Apache License 2.0 |
| `google/gemma-4-E2B-it` | Text LLM | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — **មិនមែន**ជា open source ដែលបានទទួលស្គាល់ដោយ OSI ទេ; រួមមាន Prohibited Use Policy និងលក្ខខណ្ឌកម្រិតការប្រើប្រាស់របស់ Google |
| `HuggingFaceTB/SmolVLM-256M-Instruct`, `SmolVLM-500M-Instruct` | Vision LLM | Apache License 2.0 |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Vision LLM | Apache License 2.0 |
| `openai/whisper-tiny` / `base` / `small` / `large-v3` | Speech-to-text | MIT License |
| `seanghay/whisper-small-khmer-v2` | Speech-to-text (ខ្មែរ) | សូមពិនិត្យ model card — ការកែសម្រួល Whisper ជាទូទៅទទួលមរតក MIT ប៉ុន្តែសូមផ្ទៀងផ្ទាត់ម្តងមួយៗ |
| `metythorn/whisper-large-v3-turbo-mixed-20eps-clean-text-197k` | Speech-to-text (ខ្មែរ) | សូមពិនិត្យ model card |
| `BAAI/bge-m3` | Text embeddings | MIT License |
| `vidore/colsmolvlm-v0.1` | ការទាញយកឯកសារចក្ខុវិស័យ | Apache License 2.0 |
| `vidore/colqwen2-v1.0` | ការទាញយកឯកសារចក្ខុវិស័យ | Apache License 2.0 |
| ឯកសារ `.gguf` មូលដ្ឋានណាមួយដែលអ្នកផ្តល់ | Text LLM (llama.cpp) | អាស្រ័យលើអាជ្ញាប័ណ្ណដែលអ្នកនិពន្ធម៉ូដែលបានភ្ជាប់ជាមួយឯកសារនោះ — សូមពិនិត្យ model card របស់វាមុននឹងប្រើប្រាស់ |

**Gemma គឺជាចំណុចដែលត្រូវប្រុងប្រយ័ត្នបំផុត**៖ វា *មិនមែន*ជាប្រភេទសេរីដូច
Apache/MIT ទេ។ លក្ខខណ្ឌ Gemma Terms of Use របស់ Google កំណត់ការប្រើប្រាស់
ដែលអាចទទួលយកបាន ហើយតម្រូវឱ្យអ្នកបញ្ជូនលក្ខខណ្ឌដូចគ្នានេះទៅកាន់អ្នកណាដែល
អ្នកចែកចាយម៉ូដែល ឬលទ្ធផលដែលបានមកពីវា។ ប្រសិនបើអ្នកចែកចាយកម្មវិធីនេះជាមួយ
Gemma ភ្ជាប់មកជាមួយ ឬជាលំនាំដើម សូមពិនិត្យលក្ខខណ្ឌទាំងនោះដោយខ្លួនឯង។

---

### បណ្ណាល័យ Python សំខាន់ៗ

| កញ្ចប់ | អាជ្ញាប័ណ្ណ |
|---|---|
| `smolagents` | Apache License 2.0 |
| `chromadb` | Apache License 2.0 |
| `sentence-transformers` | Apache License 2.0 |
| `transformers`, `accelerate`, `datasets` | Apache License 2.0 |
| `torch`, `torchvision`, `torchaudio` | អាជ្ញាប័ណ្ណបែប BSD |
| `llama-cpp-python` | MIT License |
| `python-docx` | MIT License |
| `PyMuPDF` (`fitz`) | **AGPL-3.0 / ពាណិជ្ជកម្ម** — សូមមើលការព្រមានខាងលើ |
| `pdf2image` | MIT License |
| `gradio` | Apache License 2.0 |
| `pandas`, `numpy` | BSD 3-Clause |
| `matplotlib` | ផ្អែកលើ PSF (matplotlib license) |
| `openpyxl` | MIT License |
| `byaldi`, `colpali-engine` | Apache License 2.0 (សូមផ្ទៀងផ្ទាត់កំណែបច្ចុប្បន្ន) |
| `soundfile`, `librosa` | BSD 3-Clause |
| `Pillow` | HPND (ប្រភេទសេរី) |

អាជ្ញាប័ណ្ណកញ្ចប់អាចផ្លាស់ប្តូររវាងកំណែផ្សេងៗគ្នា — តារាងនេះឆ្លុះបញ្ចាំង
ពីកំណែដែលបានកំណត់ក្នុង `requirements.txt` នៅពេលដែលសេចក្តីជូនដំណឹងនេះ
ត្រូវបានសរសេរ។ សម្រាប់ការត្រួតពិនិត្យពេញលេញ និងទាន់សម័យ សូមដំណើរការឧបករណ៍
ដូចជា `pip-licenses` នៅក្នុង venv របស់អ្នក។

---
---

## 🇬🇧 English

This project's own source code (`app.py`, `index_docs.py`, and related scripts)
is licensed under the **Apache License 2.0** — see [`LICENSE`](./LICENSE),
matching the license of [`smolagents`](https://github.com/huggingface/smolagents),
the agent framework this app is built on.

The Apache-2.0 license covers *this repository's code only*. It does **not**
relicense the third-party libraries and AI models the app downloads and runs
at runtime — each of those keeps its own license, which you must comply with
separately. A summary is below. **This is not legal advice** — check the
upstream source for each item if you plan to redistribute or use this
commercially.

---

## ⚠️ Important: PyMuPDF is AGPL-3.0, not permissive

`requirements.txt` includes `PyMuPDF` (imported as `fitz` in `app.py`, used
for PDF text extraction in `index_pdf_file()`). Unlike the rest of this
stack, PyMuPDF/MuPDF is dual-licensed by Artifex Software under:

- **GNU AGPL v3** (free, but copyleft — if you distribute this app or run it
  as a network service others interact with, AGPL's source-disclosure
  obligations can apply to your *combined* application), or
- **a paid commercial license from Artifex** (removes the AGPL obligations).

If you plan to keep this project closed-source, distribute it commercially,
or offer it as a hosted service, get a commercial MuPDF license from Artifex
or swap `PyMuPDF` for a permissively-licensed alternative (e.g.
`pdfplumber`, MIT-licensed, though slower). If this stays a personal/local
tool you run yourself, AGPL's obligations are far less likely to bite, but
you should still be aware of it.

---

## AI Models

Models are downloaded at runtime from Hugging Face and are **not**
redistributed as part of this repository. Each keeps its own license terms;
using the model means agreeing to those terms directly with the model owner.

| Model | Used for | License |
|---|---|---|
| `Qwen/Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B` | Text LLM | Apache License 2.0 |
| `google/gemma-4-E2B-it` | Text LLM | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — **not** OSI-approved open source; includes Google's Prohibited Use Policy and use-restriction terms |
| `HuggingFaceTB/SmolVLM-256M-Instruct`, `SmolVLM-500M-Instruct` | Vision LLM | Apache License 2.0 |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Vision LLM | Apache License 2.0 |
| `openai/whisper-tiny` / `base` / `small` / `large-v3` | Speech-to-text | MIT License |
| `seanghay/whisper-small-khmer-v2` | Speech-to-text (Khmer) | Check model card — Whisper fine-tunes typically inherit MIT, but verify per model |
| `metythorn/whisper-large-v3-turbo-mixed-20eps-clean-text-197k` | Speech-to-text (Khmer) | Check model card |
| `BAAI/bge-m3` | Text embeddings | MIT License |
| `vidore/colsmolvlm-v0.1` | Visual document retrieval | Apache License 2.0 |
| `vidore/colqwen2-v1.0` | Visual document retrieval | Apache License 2.0 |
| Any local `.gguf` file you supply | Text LLM (llama.cpp) | Whatever license the model author attached to that checkpoint — check its model card before use |

**Gemma is the one to watch**: it is *not* Apache/MIT-style permissive.
Google's Gemma Terms of Use impose acceptable-use restrictions and require
you to pass those same terms on to anyone you distribute the model or
derivatives to. If you ship this app with Gemma bundled or defaulted, review
those terms yourself.

---

## Key Python dependencies

| Package | License |
|---|---|
| `smolagents` | Apache License 2.0 |
| `chromadb` | Apache License 2.0 |
| `sentence-transformers` | Apache License 2.0 |
| `transformers`, `accelerate`, `datasets` | Apache License 2.0 |
| `torch`, `torchvision`, `torchaudio` | BSD-style license |
| `llama-cpp-python` | MIT License |
| `python-docx` | MIT License |
| `PyMuPDF` (`fitz`) | **AGPL-3.0 / commercial** — see warning above |
| `pdf2image` | MIT License |
| `gradio` | Apache License 2.0 |
| `pandas`, `numpy` | BSD 3-Clause |
| `matplotlib` | PSF-based (matplotlib license) |
| `openpyxl` | MIT License |
| `byaldi`, `colpali-engine` | Apache License 2.0 (verify current version) |
| `soundfile`, `librosa` | BSD 3-Clause |
| `Pillow` | HPND (permissive) |

Package licenses can change between versions — this table reflects the
versions pinned in `requirements.txt` at the time this notice was written.
For a full, current audit, run something like `pip-licenses` in your venv.
