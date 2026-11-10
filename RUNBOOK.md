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

## 11. Follow-up run: Mistral 7B, Llama 3.1 8B and Qwen2.5-14B with quantized setups

vLLM 0.33 dropped bitsandbytes, and dynamic FP8 does not load on an A100, so the first run gave Mistral only BF16 and
the FP16 control. This run adds quantized setups for three families on one GPU with native FP8:

| Family | Setups |
|---|---|
| Mistral 7B | BF16, FP16, FP8, INT8 weight-only, INT4 GPTQ built with llm-compressor, FP8 KV cache |
| Llama 3.1 8B | BF16, FP16, FP8, INT8 W8A16, INT8 weight-only, INT4 GPTQ, INT4 AWQ, FP8 KV cache |
| Qwen2.5-14B | BF16, FP16, FP8, INT8 GPTQ, INT4 GPTQ, INT4 AWQ, FP8 KV cache, from Qwen's official checkpoints |

It also runs RUT on Mistral's new setups. Qwen2.5-14B shows detection on a model twice the size of the others, and Muse
Glimmer keeps its FP16 control results from the A100 at 30B.

**GPU choice.** The job needs native FP8, so it runs on Ada, Hopper and Blackwell GPUs and stops at once on an A100,
A6000 or A40. Pick the first that RunPod has available:

| GPU | Memory | Families | Rough cost |
|---|---|---|---|
| L40S, L40 or RTX 6000 Ada | 48 GB | All three | About $4 to $5 |
| **RTX PRO 6000** | **96 GB** | All three, and optionally quantized Muse Glimmer | About $7 to $10, or about $13 to $16 with Muse |
| H100 | 80 GB | All three, and optionally Muse | About $10 to $12 |
| RTX 4090 or 5090 | 24 or 32 GB | Mistral and Llama only | About $3 to $4 |

Check the hourly price on the deploy page, since these are estimates. If a GPU shows no capacity, set Region to
"Any" and try both Secure Cloud and Community Cloud.

**Optional, on an RTX PRO 6000 or H100: add quantized Muse Glimmer.** Run this line before starting the job. It adds
about 3 hours and needs a **250 GB** disk:

```bash
export FAMILIES="mistral-7b llama3.1-8b qwen2.5-14b muse-glimmer-30b"
```

**Accept the Llama 3.1 8B license first** at `huggingface.co/meta-llama/Llama-3.1-8B-Instruct`. The job runs Llama
second, so approval has time to arrive. Without it, the job skips Llama and runs the rest.

1. Push the latest scripts from your laptop, the same way as in step 0.
2. Deploy your GPU, On-Demand, RunPod PyTorch, with a **150 GB** container disk, or **250 GB** if you add Muse, and
   at least **$15** of credit, or **$25** with Muse.
3. On the pod, one line at a time:

```bash
cd ~ && git clone -q https://github.com/prithsk/quantprint-experiments && cd quantprint-experiments && ls jobs
export HF_TOKEN=hf_YOUR_TOKEN_HERE
export REPEATS=3
bash run_gpu.sh jobs/h100_families.sh
```

About 5 hours. Data lands in folders named after the GPU, such as `data-qwen2.5-14b-l40s`. A setup that fails to load
is skipped and logged, so paste any SKIP or FAILED lines to Claude. When the log lists the results files, download every
`results-*.tar.gz` file it names, terminate the pod, push them with the step 8 commands and tell Claude.

**After this run, the audit can cover Llama too.** Run `python audit_openrouter.py --model meta-llama/llama-3.1-8b-instruct --list`
to see its providers, run it again without `--list`, and score them with `python audit_score.py --ref data-llama3.1-8b-l40s --audit data-audit-llama-3.1-8b-instruct`,
using the folder name for the GPU you rented.

## 12. Protecting results on long runs

A pod that stops, for example when the balance runs out, loses its whole disk. Two safeguards prevent that:

1. **In RunPod, turn on auto-pay and a low-balance alert** under Billing before any long run.
2. **Use a Hugging Face token with WRITE access as HF_TOKEN.** The long jobs then upload each finished family's results
   to a private dataset, `huggingface.co/datasets/prithsk/quantprint-results`, right after the family finishes, so a
   stopped pod loses at most the family in progress. Create the token under Settings, Access Tokens, and choose
   **Write**. Watch the log for `uploaded results-...` lines to confirm it works. On your laptop,
   `python download_results.py` downloads and unpacks every uploaded file into the repo.
