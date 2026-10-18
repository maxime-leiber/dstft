## Summary

<!--
What changed and why. Link the issue/discussion this addresses if there is
one. For a fix, include the root cause, not just the symptom.
-->

## Test plan

<!--
Checklist of what you ran to verify this, e.g.:
- [ ] `pytest` / `pytest --cov=dstft --cov-report=term-missing`
- [ ] `pre-commit run --all-files`
- [ ] `sphinx-build -b html docs docs/_build/html -W --keep-going` (if docs changed)
- [ ] Manual verification (describe what you checked)
-->

<!--
If this PR changes numerical behavior in `src/dstft/` (a new formula, a new
mode, a changed default, a bug fix that alters output), say so explicitly
and include a comparison against the previous behavior (e.g. a gradcheck or
a torch.stft parity check) — see CONTRIBUTING.md.
-->
