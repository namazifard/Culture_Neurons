#!/usr/bin/env bash
set -euo pipefail

# 1) System deps (skip if already installed)
PKGS=( make build-essential libssl-dev zlib1g-dev libbz2-dev \
       libreadline-dev libsqlite3-dev wget curl llvm \
       libncursesw5-dev xz-utils tk-dev libxml2-dev \
       libxmlsec1-dev libffi-dev liblzma-dev )
MISSING=()
for pkg in "${PKGS[@]}"; do
  if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null \
       | grep -q "ok installed"; then
    MISSING+=( "$pkg" )
  fi
done
if (( ${#MISSING[@]} )); then
  echo "Installing: ${MISSING[*]}"
  sudo apt update
  sudo apt install -y "${MISSING[@]}"
else
  echo "✓ All apt packages present"
fi

# 2) Bootstrap pyenv if needed
if [ -d "$HOME/.pyenv" ]; then
  echo "✓ pyenv already installed"
else
  echo "Installing pyenv…"
  curl https://pyenv.run | bash
fi

# 3) Shell setup for pyenv (current session)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# 4) Persist into ~/.bashrc
if ! grep -q 'pyenv init --path' ~/.bashrc; then
  cat >> ~/.bashrc <<EOF

# >>> pyenv setup >>>
export PYENV_ROOT="\$HOME/.pyenv"
export PATH="\$PYENV_ROOT/bin:\$PATH"
eval "\$(pyenv init --path)"
eval "\$(pyenv init -)"
# <<< pyenv setup <<<
EOF
  echo "Appended pyenv init to ~/.bashrc"
else
  echo "✓ pyenv init already in ~/.bashrc"
fi

# 5) Python & project
pyenv install -s 3.10.13
mkdir -p /work/LAPE
cd /work/LAPE
pyenv local 3.10.13

# 6) GPU-enabled torch + vllm (only if missing)
if ! pip show torch >/dev/null; then
  pip install https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp310-cp310-linux_x86_64.whl
else
  echo "✓ torch already installed"
fi

if ! pip show vllm >/dev/null; then
  pip install vllm==0.2.7
else
  echo "✓ vllm already installed"
fi

# 7) Additional Python packages
pip install pandas matplotlib tabulate
pip install "numpy<2"

echo "✅ setup complete — open a new shell or run 'source ~/.bashrc' to pick up pyenv."
