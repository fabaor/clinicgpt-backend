# Appium: continuação da sessão

## Objetivo
Instalar o Appium (https://appium.io/docs/en/latest/) com todos os recursos e usar o Appium Inspector com Chrome no Mac.

## Ambiente do usuário
- macOS (MacBook Pro), shell zsh, usuário `fabiorieger`.
- Appium instalado em `~/.npm-global` (prefixo npm do usuário, sem sudo). Isso corrigiu o erro `EACCES` do `npm install -g`.
- Script de instalação: `setup-appium.sh` (nesta branch) ou `~/setup-appium.sh`.

## Já instalado (confirmado rodando: `GET /status 200`)
- Appium 4.0.1.
- Drivers: uiautomator2, espresso, safari, gecko, chromium, xcuitest, mac2, wincore.
- Plugins: execute-driver, images, inspector, relaxed-caps, storage, universal-xml.

## Fatos importantes
- No Appium 4.0.1 NÃO existe `--allow-cors` (dá "Unrecognized arguments"). Não há opção de CORS.
- Iniciar com Inspector: `appium --use-plugins=inspector`.
- O servidor leva ~10 s para subir. Esperar `listener started` antes de abrir o navegador.
- Inspector do plugin: http://127.0.0.1:4723/inspector (testado na sessão de nuvem: HTTP 200, plugin ACTIVE).
- Alternativa sem CORS: `brew install --cask appium-inspector`.
- `wincore` só funciona no Windows. `xcuitest`, `mac2` e Safari exigem Xcode/macOS.

## Estado atual / pendências
1. O usuário recebeu o erro "Could not connect to Appium server URL 'http://127.0.0.1:4723'" no Inspector (provável: servidor ainda subindo ou Inspector hospedado na web).
2. Ainda não testou uma sessão de Chrome no Mac. Capabilities:
   `{"platformName":"mac","browserName":"chrome","appium:automationName":"chromium"}`
3. Possível erro seguinte: chromedriver incompatível com a versão do Chrome. Se ocorrer, pedir a mensagem completa do log.
4. Android: falta Android Studio / SDK e `ANDROID_HOME`. iOS: falta Xcode e simulador.
5. O usuário quer que o Claude opere o Chrome via extensão Claude in Chrome. Isso só funciona numa sessão local (Claude Desktop ou Claude Code no Mac), não em sessão na nuvem.

## Próximos passos sugeridos
1. Rodar `appium --use-plugins=inspector` e confirmar com `curl http://127.0.0.1:4723/status`.
2. Abrir o Inspector e conectar (Host 127.0.0.1, Port 4723, Path /) com as capabilities do Chrome.
3. Resolver erros de chromedriver, se houver.
4. Configurar Android e/ou iOS, conforme a escolha do usuário.
