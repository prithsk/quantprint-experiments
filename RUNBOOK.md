# Runbook: every remaining experiment for the ICML version

Run the steps in order. Each GPU step prints its own result table at the end. Paste every table to Claude, and Claude
updates the paper. Nothing here needs code changes.

## 0. One-time setup

**First, push the new scripts to GitHub.** The pod clones the repo, so it only sees what GitHub has. On your laptop:

```powershell
Set-Location "$HOME\ship\quantprint-experiments"
$dl = Get-ChildItem "$HOME\Downloads\quantprint-update3*" | Sort-Object LastWriteTime | Select-Object -Last 1
tar -xzf $dl.FullName -C .
git add -A; git commit -q -m "scripts for the remaining ICML experiments"; git push -q
git log -1 --format="%s"
```

The last line must print "scripts for the remaining ICML experiments". On the pod, `ls jobs` must then list
`all.sh`, `family.sh`, `h100_quant.sh`, `rut.sh` and `sglang.sh`.

**Hugging Face token.** Every model in the plan is open, so no license approval is needed: Qwen and Muse Glimmer use
Apache 2.0, and the Mistral page shows no access form. A token still makes downloads faster and avoids rate limits.
1. Make a free account at huggingface.co.
2. Click your profile picture, then **Settings, Access Tokens, Create new token.** Choose **Read**, create it and
   copy it right away, since Hugging Face shows it only once. It starts with `hf_`.

**RunPod pod** (same steps as the H100 run):
1. Deploy an **H100** or **A100 80 GB**, **On-Demand**, with the **RunPod PyTorch** template.
2. Set **Container Disk to 250 GB.** The 32B model and its quantized versions need room.
3. Open the web terminal and run these two lines, one at a time, with your real token:

```bash
cd ~ && git clone -q https://github.com/prithsk/quantprint-experiments && cd quantprint-experiments && ls jobs
export HF_TOKEN=hf_YOUR_TOKEN_HERE
export REPEATS=3
```

`REPEATS` sets how many times each setup runs at each batch size. The paper so far uses 2. Three gives tighter
intervals for ICML and adds about half the collection time; leave the line out to keep 2. The RUT job draws its own
samples and ignores this setting.

**Getting results off the pod:** every job saves a `results-*.tar.gz` file in `/workspace`. Open Jupyter Lab from
the pod's Connect menu and download it from the file panel. If Jupyter does not open, run
`curl -s -F "file=@/workspace/results-NAME.tar.gz" https://0x0.st` and open the link it prints.
**Terminate the pod** once every file is downloaded.

## The fastest path: everything at once

```bash
bash run_gpu.sh jobs/all.sh
```

This runs steps 1 to 5 back to back in about 5 hours with 2 repeats, roughly $20 to $25 on an H100, or about 7 hours
and $30 with 3 repeats. The log shows a table after
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

## 2. Muse Glimmer: the second family and the larger model

```bash
bash run_gpu.sh jobs/family.sh muse-glimmer-30b
```

About 60 to 75 minutes, and it needs an 80 GB GPU. Muse Glimmer is Meta's open model from August 2026, about 30B
parameters under Apache 2.0, so it covers both a second model family and a larger model. Meta ships no GPTQ or AWQ
versions of it, so its quantized setups are vLLM's own FP8 and bitsandbytes 4-bit, next to BF16 and the FP16 control.
The job first checks that the model exists on Hugging Face. A setup that fails to load on this new architecture is
skipped and logged, and only a failed BF16 reference stops the job. Paste any SKIP message to Claude.

## 3. Mistral 7B: the comparison on RUT's own model

```bash
bash run_gpu.sh jobs/family.sh mistral-7b
bash run_gpu.sh jobs/rut.sh mistral-7b
```

About 40 minutes each. Zhu et al. tested RUT on Mistral 7B, so this compares both tests on their own model. The family
job must finish first, since the RUT comparison reads its data.

**Optional extra families.** Llama 3.1 8B, with GPTQ and AWQ versions, and Qwen 2.5 32B stay available. Run
`bash run_gpu.sh jobs/family.sh llama3.1-8b`, or set `export EXTRA_FAMILIES="llama3.1-8b"` before `jobs/all.sh`.
Llama is gated, so accept its license on Hugging Face first.

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
python audit_score.py --ref data-nocache --audit data-audit-qwen-2.5-7b-instruct
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

## 9. Submission logistics

ICML has used OpenReview for submissions, so start these early. Confirm each rule in the ICML 2027 call for papers,
since rules change from year to year.

- **OpenReview profile:** create one now with your utexas.edu email. New profiles go through moderation, which can
  take days, and an institutional email speeds it up.
- **Reviewing duty:** recent ICML cycles asked qualified authors to review. Check what the 2027 call says.
- **LLM policy:** recent cycles have published rules on using language models in writing. Read the 2027 rules.
- **Anonymity:** when you create the anonymous code link, add `prithsk`, `Krishnan` and `utexas` as terms to hide.
  If you post a preprint, check the 2027 rules on preprints first.
- **Supplementary material:** upload the anonymized code as a zip if the call allows it, so reviewers can rerun it.
- **Page limit:** the main body must fit in 8 pages, and the impact statement, references and appendix do not count.

## 10. Paper revisions after the results come in

Paste each results table to Claude. Claude then:
- adds the RUT comparison, the new families, the 32B model, the quantized H100 runs, SGLang and the audit;
- replaces every red placeholder;
- adds related work that the new results touch, including the prompt caching audit of Gu et al. (2025), which
  relates to the prefix caching finding, model idiosyncrasies of Sun et al. (2025) and LLMmap of Pasquini et al.
  (2024);
- reruns the style scan and the page check, and confirms that every number traces back to data on GitHub.

## 11. Follow-up run: Llama 3.1 8B, Mistral and Muse Glimmer on an H100

vLLM 0.33 dropped bitsandbytes, and dynamic FP8 does not load on an A100, so the first run gave Mistral and Muse only
BF16 and the FP16 control. This run collects every family below on one H100, in folders ending in `-h100`:

| Family | Setups |
|---|---|
| Llama 3.1 8B | BF16, FP16, FP8, INT8 W8A16, INT8 weight-only, INT4 GPTQ, INT4 AWQ, FP8 KV cache |
| Mistral 7B | BF16, FP16, FP8, INT8 weight-only, INT4 GPTQ built with llm-compressor, FP8 KV cache |
| Muse Glimmer 30B | BF16, FP16, FP8, INT8 weight-only, INT4 GPTQ built with llm-compressor, FP8 KV cache |

- **Llama 3.1 8B** gives the provider audit a widely served model. Its INT4 checkpoints come from hugging-quants.
- **INT4 GPTQ for Mistral and Muse** comes from `quantize.py`, calibrated on Dolly filler prompts, never on the probes.
- **The FP8 KV cache** keeps BF16 weights and stores the attention cache in FP8, a common provider optimization.
- The new INT8, INT4 and KV-cache setups compute in BF16, so they also test the paper's mechanism.
- The job runs RUT on Mistral's new setups too.

**Before the run, accept the Llama 3.1 8B license** at huggingface.co on the `meta-llama/Meta-Llama-3.1-8B-Instruct`
page. Approval usually takes under an hour. Without it, the job skips Llama and runs everything else.

1. Push the latest scripts from your laptop, the same way as in step 0.
2. Deploy an **H100**, On-Demand, RunPod PyTorch, **250 GB** container disk, with at least **$35** of credit.
3. On the pod, one line at a time:

```bash
cd ~ && git clone -q https://github.com/prithsk/quantprint-experiments && cd quantprint-experiments && ls jobs
export HF_TOKEN=hf_YOUR_TOKEN_HERE
export REPEATS=3
bash run_gpu.sh jobs/h100_families.sh
```

About 5 to 6 hours. The job stops at once on any GPU other than an H100. A setup that fails to load is skipped and
logged, so paste any SKIP or FAILED lines to Claude. When the log lists the results files, download every
`results-*-h100.tar.gz` file, terminate the pod, push them with the step 8 commands and tell Claude.

**After this run, the audit can cover Llama too.** Run `python audit_openrouter.py --model meta-llama/llama-3.1-8b-instruct --list`
to see its providers, run it again without `--list`, and score them with
`python audit_score.py --ref data-llama3.1-8b-h100 --audit data-audit-llama-3.1-8b-instruct`.
