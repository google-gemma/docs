# Work in progress

Temporary tracking file. **Delete before the final commit.**

## 1. Quick fixes
- [x] `components.md` prompt tokens: colour every token, drop the `\|` escaping
- [x] Card/tile surface -> `#f7f9fc` in light mode (deployment, quickstart, learn, community)
- [x] QAT page: 26B A4B missing from the availability table
- [x] vLLM: `vllm serve google/gemma-4-12B-it-qat-w4a16-ct` block not marked `.model-swap`

## 2. Model selector rollout
- [x] `deployment/lmstudio.md`
- [x] `deployment/keras.md` (checkpoints from Kaggle keras/gemma4)
- [x] `deployment/litert.md` (retarget to Gemma 4)
- [x] `deployment/gemini-api.md` (tabs per language; only 31B + 26B A4B supported)
- [x] `deployment/langchain.md`

## 3. Model data
- [x] Expand `data/models.yml` with sizes/ids for every family
- [x] Validate against the in-repo model cards first, then the collection pages
- [x] Do not invent: omit anything not attested

## 4. Structure
- [x] Remove `deployment/web.md` + `deployment/mobile.md`
- [x] Add AI Edge Gallery page
- [x] `deployment/google-cloud.md`: grid overview like the Run overview

## 5. External links
- [x] Full sweep: tag every generation-pinned external URL

## Verification (run before each commit)
```
uv run --frozen python -m unittest discover -s tests -p "test_*.py"
uv run --frozen python scripts/lint_docs.py --baseline
uv run --frozen python scripts/build_docs.py
```
