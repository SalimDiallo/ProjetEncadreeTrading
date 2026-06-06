# ⚡ Quickstart Docker (5 minutes)

Guide express pour lancer le projet dans Docker depuis zéro.

---

## 🪟 Windows

### 1. Installer Docker Desktop

1. Télécharger : <https://www.docker.com/products/docker-desktop/>
2. Lancer l'installateur, cocher **"Use WSL 2 instead of Hyper-V"** si demandé
3. Redémarrer le PC
4. Lancer **Docker Desktop** depuis le menu Démarrer
5. Attendre que l'icône baleine 🐳 en bas à droite passe au vert

### 2. Installer VS Code + extension Dev Containers

1. VS Code : <https://code.visualstudio.com/>
2. Dans VS Code → Extensions (`Ctrl+Maj+X`) → chercher **"Dev Containers"** → Install

### 3. Cloner / extraire le projet

Soit avec Git :
```powershell
git clone https://github.com/SalimDiallo/ProjetEncadreeTrading.git
cd ProjetEncadreeTrading
code .
```

Soit en décompressant le ZIP, puis :
```powershell
cd C:\Users\TonNom\Downloads\projetTrading_final
code .
```

### 4. Ouvrir dans le conteneur

Dès que VS Code ouvre le dossier, **une notification apparaît en bas à droite** :

> 📦 *Folder contains a Dev Container configuration file. Reopen in container?*  
> **[Reopen in Container]**

Clique. ☕ Attends 3-5 minutes (premier build).

### 5. C'est prêt !

Dans le terminal de VS Code (qui est maintenant DANS le conteneur) :

```bash
# Lancer le dashboard
cd web
streamlit run app.py
```

VS Code affiche une popup :
> *Your application running on port 8501 is available.*  
> **[Open in Browser]**

Clique. 🎉 **Le dashboard est ouvert**.

---

## 🍎 Mac

Pareil que Windows, sauf l'étape 1 :

1. Télécharger Docker Desktop pour Mac : <https://www.docker.com/products/docker-desktop/>
   - **Apple Silicon (M1/M2/M3/M4)** : version ARM64
   - **Intel** : version x86_64

Le reste est identique.

---

## 🐧 Linux

### 1. Installer Docker
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker      # ou redémarre la session
```

### 2. Installer VS Code + Dev Containers
```bash
# VS Code via apt (Ubuntu/Debian)
sudo snap install code --classic
# Puis installer l'extension Dev Containers depuis VS Code
```

### 3. Reste identique
```bash
cd projetTrading_final
code .
# Puis "Reopen in Container" dans VS Code
```

---

## ✅ Vérifications rapides

Si tout fonctionne, ces commandes doivent toutes réussir :

```bash
# Sur ton hôte
docker --version           # Docker version 24.x.x
docker compose version     # Docker Compose v2.x.x

# Dans le conteneur (terminal VS Code après Reopen in Container)
python --version           # Python 3.12.x
streamlit --version        # Streamlit, version 1.3x.x
pytest --version           # pytest 7.x.x
```

---

## 🆘 En cas de problème

| Erreur | Solution |
|---|---|
| "Cannot connect to Docker daemon" | Lance Docker Desktop |
| "Port 8501 already in use" | Une autre app utilise le port. Tue-la ou change le port dans `docker-compose.yml` |
| "WSL 2 installation incomplete" (Windows) | <https://aka.ms/wsl2kernel> → installer le package puis redémarrer |
| Build très lent | Normal la 1re fois. Les fois suivantes utilisent le cache. |
| "no space left on device" | `docker system prune -a` pour libérer de l'espace |

---

## 🎬 Tu fais ça une fois, ensuite c'est juste...

Pour ouvrir le projet les jours suivants :
1. Lancer Docker Desktop
2. Ouvrir VS Code → Open Folder → projetTrading
3. "Reopen in Container" (ou ça se fait automatiquement)
4. Terminal → `cd web && streamlit run app.py`
5. Coder normalement 🚀

**Avantages** :
- 🔒 Zéro pollution de ton système Python local
- 🌍 Identique sur Windows, Mac, Linux
- 👥 Tes coéquipiers ont la **même version exacte** que toi
- 📦 Plus de "ça marche chez moi" — ça marche partout pareil
