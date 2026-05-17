# Leçons

[2026-05-17] | `source .venv/bin/activate` ne s'activait pas (VIRTUAL_ENV vide, python = pyenv) | Cause : `pyenv virtualenv-init` dans ~/.bashrc met un hook PROMPT_COMMAND qui repousse les shims pyenv en tête du PATH à chaque prompt, écrasant le venv. Règle : avec un projet uv, préférer `uv run` ; si activation manuelle requise, vérifier que pyenv virtualenv-init ne hijacke pas le PATH.
