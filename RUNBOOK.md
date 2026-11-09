# Runbook: every remaining experiment for the ICML version

Run the steps in order. Each GPU step prints its own result table at the end. Paste every table to Claude, and Claude
updates the paper. Nothing here needs code changes.

## 0. One-time setup

**Hugging Face token** (needed for Llama and Mistral, which are gated):
1. Make a free account at huggingface.co.
2. Open `meta-llama/Meta-Llama-3.1-8B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3` and accept each license.
   Meta usually approves within an hour.
3. Go to Settings, then Access Tokens, and create a **Read** token. It starts with `hf_`.

**RunPod pod** (same steps as the H100 run):
1. Deploy an **H100** or **A100 80 GB**, **On-Demand**, with the **RunPod PyTorch** template.
2. Set **Container Disk to 250 GB.** The 32B model and its quantized versions need room.
3. Open the web terminal and run these two lines, one at a time, with your real token:

```bash
cd ~ && git clone -q https://github.com/prithsk/quantprint-experiments && cd quantprint-experiments && ls jobs
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

**Getting results off the pod:** every job saves a `results-*.tar.gz` file in `/workspace`. Open Jupyter Lab from
the pod's Connect menu and download it from the file panel. If Jupyter does not open, run
`curl -s -F "file=@/workspace/results-NAME.tar.gz" https://0x0.st` and open the link it prints.
**Terminate the pod** once every file is downloaded.

## The fastest path: everything at once

```bash
bash run_gpu.sh jobs/all.sh
```

This runs steps 1 to 5 back to back in about 5 to 6 hours, roughly $20 to $25 on an H100. The log shows a table after
each step, and a failed step does not stop the rest. If the tab disconnects, reconnect and run
`tail -F ~/quantprint-experiments/logs/all.log`.

To run one step at a time instead, use the command under each step.

## 1. RUT baseline, the most important step

Zhu et al. have not released their code, so `rut.py` reimplements RUT from their paper with their exact settings:
temperature 0.5, 30 tokens, one target and 100 reference responses per prompt, the log-rank score and a Cramer-von
Mises test.

```bash
bash run_gpu.sh jobs/rut.sh qwen2.5-7b
```

About 45 to 60 minutes. The table puts RUT and QuantPrint side by side at 5% and 1% false positive rates.
**This decides the venue.** If QuantPrint clearly beats RUT on FP16 and INT8, stay with ICML. If RUT matches it,
aim for COLM, where the attribution and hardware results still stand.

## 2. Two more model families

```bash
bash run_gpu.sh jobs/family.sh llama3.1-8b
bash run_gpu.sh jobs/family.sh mistral-7b
```

About 40 minutes each. Each job first checks that every model exists and that your token can reach it, and it skips
any setup it cannot reach. Mistral 7B is the model Zhu et al. tested RUT on, so also run
`bash run_gpu.sh jobs/rut.sh mistral-7b` for a comparison on their own model.

## 3. A larger model

```bash
bash run_gpu.sh jobs/family.sh qwen2.5-32b
```

About 60 to 90 minutes. This needs an 80 GB GPU.

## 4. Quantized models on another GPU

```bash
bash run_gpu.sh jobs/h100_quant.sh
```

About 25 minutes. It scores quantized Qwen 7B runs from this GPU against the L4 fingerprints, which is the half of
the hardware test the paper still lacks.

## 5. SGLang as a second engine

```bash
bash run_gpu.sh jobs/sglang.sh
```

About 60 minutes. It installs SGLang in its own environment and runs one server per setup with prefix caching off.
If it stops with "Unexpected SGLang response format", paste the printed response to Claude.

## 6. Provider audit, on your laptop

1. Make an account at openrouter.ai, add $5 of credit and create an API key.
2. In PowerShell:

```powershell
Set-Location "$HOME\ship\quantprint-experiments"
Add-Content .env "OPENROUTER_API_KEY=sk-or-YOUR_KEY_HERE"
python audit_openrouter.py --list
python audit_openrouter.py
python audit_score.py
```

The first command only lists each provider, its declared precision and whether it returns log-probabilities. The
full run costs well under a dollar. A provider gets no verdict when too few of its responses start with the same
tokens as the reference, which signals a different chat template, so it never gets a false accusation.
**Before the paper names any provider, email that provider the result and give it a chance to respond.**

## 7. Paperwork

- **Anonymous code link:** go to anonymous.4open.science, sign in with GitHub, choose `prithsk/quantprint-experiments`
  and create the link. Paste it to Claude for the Software and Data section.
- **ICML 2027 style files:** when ICML posts them, replace `icml2026.sty` and `icml2026.bst` and change the two
  matching lines in `main.tex`.
- **Deadline:** check icml.cc for the 2027 call for papers. Register the abstract before its deadline, since a missed
  abstract usually blocks the full submission.
- **Faculty review:** ask one UT faculty member who works on machine learning or systems to read the draft.

## 8. Saving results to GitHub

After each download, on your laptop:

```powershell
Set-Location "$HOME\ship\quantprint-experiments"
$t = Get-ChildItem "$HOME\Downloads\results-*.tar.gz" | Sort-Object LastWriteTime | Select-Object -Last 1
$t.FullName
tar -xzf $t.FullName -C .
git add data-* results; git commit -q -m "results: $($t.BaseName)"; git push -q
```
