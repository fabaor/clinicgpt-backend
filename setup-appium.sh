#!/usr/bin/env bash
# Instala o Appium com todos os drivers e plugins oficiais, sem sudo.
set -euo pipefail

# 1) npm global dentro do seu usuário (evita EACCES em /usr/local)
mkdir -p "$HOME/.npm-global"
npm config set prefix "$HOME/.npm-global"
export PATH="$HOME/.npm-global/bin:$PATH"

# Persiste o PATH no shell (zsh é o padrão no macOS)
RC="$HOME/.zshrc"; [ -n "${BASH_VERSION:-}" ] && [ -f "$HOME/.bashrc" ] && RC="$HOME/.bashrc"
grep -q '.npm-global/bin' "$RC" 2>/dev/null || echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$RC"

# 2) Appium
npm install -g appium
appium --version

# 3) Drivers (os que não se aplicam ao seu SO podem falhar; seguimos adiante)
for d in uiautomator2 espresso safari gecko chromium xcuitest mac2 wincore; do
  appium driver install "$d" || echo "AVISO: driver $d não instalado"
done

# 4) Plugins
for p in execute-driver images inspector relaxed-caps storage universal-xml; do
  appium plugin install "$p" || echo "AVISO: plugin $p não instalado"
done

echo
appium driver list --installed
appium plugin list --installed
echo "Pronto. Abra um novo terminal (ou rode: source $RC) e execute: appium"
