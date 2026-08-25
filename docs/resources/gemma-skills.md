---
title: Gemma Skills
description: Agent skills that teach a coding agent how to build with Gemma and how to fine-tune it.
models: [gemma-4]
---

# Gemma Skills

[Gemma Skills][gemma-skills-repo] are installable skills that give a
coding agent working knowledge of Gemma — current model names, the APIs to reach for, and the
steps a task actually takes. Install one and the agent stops guessing.

<div class="card-grid">
<a class="card" href="https://github.com/google-gemma/gemma-skills/tree/main/skills" target="_blank" rel="noopener">
  <div class="card-title">gemma-dev <span class="card-ext">↗</span></div>
  <p class="card-desc">Building applications with Gemma, and general questions about the models.</p>
</a>
<a class="card" href="https://github.com/google-gemma/gemma-skills/tree/main/skills" target="_blank" rel="noopener">
  <div class="card-title">gemma-trainer <span class="card-ext">↗</span></div>
  <p class="card-desc">Training and adapting Gemma on local hardware — SFT, DPO, RLHF and reward modeling.</p>
</a>
</div>

## Install

=== "skills CLI"

    ```bash
    npx skills add google-gemma/gemma-skills --skill gemma-dev --global
    ```

=== "Context7 CLI"

    ```bash
    npx ctx7 skills install /google-gemma/gemma-skills gemma-dev
    ```

Swap `gemma-dev` for `gemma-trainer`, or browse them interactively with
`npx skills add google-gemma/gemma-skills --list`.

!!! note
    The repository is Apache-2.0 licensed and is not an officially supported Google product.
