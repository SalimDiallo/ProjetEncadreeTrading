# 🐳 Guide Docker — Petrol Trading Platform

> Tout faire en conteneur Docker sans polluer ta machine locale.
> Aucune installation Python sur ton PC nécessaire (sauf Docker lui-même).

---

## 📋 Pré-requis

Une seule chose à installer sur ta machine :

### Windows / Mac
- **Docker Desktop** : <https://www.docker.com/products/docker-desktop/>

### Linux
- Docker Engine : `sudo apt install docker.io docker-compose-plugin`
- Ajouter ton user au groupe docker : `sudo usermod -aG docker $USER` (puis redémarre)

### Vérifier l'installation
```bash
docker --version              # Docker version 24.x ou +
docker compose version        # Docker Compose v2.x ou +
```

---

## 🚀 Démarrage rapide (3 commandes)

```bash
# 1. Construire l'image (la première fois seulement, ~5 min)
docker compose build

# 2. Lancer le dashboard
docker compose up dashboard

# 3. Ouvrir dans le navigateur
# → http://localhost:8501
```

C'est tout ! Le dashboard tourne dans un conteneur, isolé de ton système.

Pour arrêter : `Ctrl+C` dans le terminal, puis `docker compose down`.

---

## 🎯 Méthode recommandée : Dev Container dans VS Code

Cette méthode te permet d'**ouvrir VS Code dans le conteneur** : tu codes
normalement, mais Python, pip et toutes les libs sont dans Docker.

### Installation (une seule fois)

1. **Installer l'extension VS Code "Dev Containers"** :
   - Ouvre VS Code
   - Va dans Extensions (`Ctrl+Maj+X`)
   - Cherche **"Dev Containers"** (de Microsoft)
   - Clique sur **Install**

### Ouverture du projet en mode conteneur

1. **Ouvre le dossier du projet** dans VS Code (`File > Open Folder...`)

2. VS Code détectera automatiquement le `.devcontainer/devcontainer.json`
   et affichera une notification en bas à droite :
   > 📦 *Folder contains a Dev Container configuration file.*  
   > **Reopen in Container**

3. Clique sur **"Reopen in Container"**.

   *(Sinon : `Ctrl+Maj+P` → tape `Dev Containers: Reopen in Container`)*

4. **Première fois : VS Code construit l'image** (~5 minutes).
   Tu peux suivre la progression en cliquant sur "Starting Dev Container".

5. Quand c'est prêt, **VS Code est maintenant DANS le conteneur**.
   Tu peux le voir en bas à gauche : `>< Dev Container: Petrol Trading Platform`.

### Que se passe-t-il maintenant ?

- ✅ Le terminal intégré (`Ctrl+ù`) est celui du conteneur
- ✅ Python pointe sur celui du conteneur (3.12)
- ✅ Toutes les libs (streamlit, pandas, pytest...) sont déjà installées
- ✅ Les ports 8501 (Streamlit) et 8888 (Jupyter) sont forwardés automatiquement
- ✅ Les changements de code sont reflétés en temps réel (volume monté)

### Lancer le dashboard depuis VS Code

Dans le terminal intégré :
```bash
cd web
streamlit run app.py
```

VS Code te propose automatiquement d'ouvrir `http://localhost:8501` 🎉.

### Lancer les tests
```bash
cd web
pytest
```

---

## 🛠️ Toutes les commandes utiles

### Build & Run

| Commande | Action |
|---|---|
| `docker compose build` | Reconstruit l'image (après changement Dockerfile) |
| `docker compose up dashboard` | Lance le dashboard Streamlit (port 8501) |
| `docker compose up jupyter` | Lance Jupyter Lab (port 8888) |
| `docker compose up -d dashboard` | Lance en arrière-plan (-d = detached) |
| `docker compose down` | Arrête et supprime les conteneurs |
| `docker compose logs -f dashboard` | Voir les logs en direct |

### Exécuter des commandes ponctuelles

| Commande | Action |
|---|---|
| `docker compose run --rm test` | Lance les tests pytest |
| `docker compose run --rm shell` | Ouvre un terminal bash interactif |
| `docker compose run --rm scraping` | Exécute le pipeline scraping |

### Gestion des images

| Commande | Action |
|---|---|
| `docker images` | Liste les images sur ta machine |
| `docker rmi petrol-trading:latest` | Supprime l'image |
| `docker system prune` | Nettoie les ressources inutilisées |

---

## 📁 Architecture Docker

```
projetTrading/
├── Dockerfile                  ← Définition de l'image
├── docker-compose.yml          ← Orchestration des services
├── .dockerignore              ← Fichiers exclus du build
│
└── .devcontainer/             ← Config VS Code Dev Container
    └── devcontainer.json
```

### Services définis dans `docker-compose.yml`

```
dashboard  → Streamlit (8501)        [service principal]
jupyter    → Jupyter Lab (8888)
test       → pytest                  [profile: test]
shell      → bash interactif         [profile: shell]
scraping   → pipeline de scraping    [profile: scraping]
```

---

## 🐛 Dépannage

### "Cannot connect to the Docker daemon"
Docker Desktop n'est pas lancé. Démarre-le depuis le menu Start (Windows)
ou Applications (Mac).

### Port 8501 déjà utilisé
Une autre app utilise le port. Soit tu la tues, soit tu changes le port :

Dans `docker-compose.yml` :
```yaml
ports:
  - "8502:8501"   # 8502 sur ta machine → 8501 dans le conteneur
```

### Le code change mais le dashboard ne se met pas à jour
Streamlit a un mode auto-reload, mais parfois il faut rafraîchir le navigateur
(`Ctrl+R`). Si ça ne marche toujours pas, redémarre le conteneur :
```bash
docker compose restart dashboard
```

### "permission denied" lors de l'écriture de fichiers
Sur Linux, l'utilisateur dans le conteneur est root. Les fichiers créés
appartiendront à root sur ton hôte. Solutions :
- Soit travailler exclusivement via le Dev Container (recommandé)
- Soit ajuster les permissions : `sudo chown -R $USER:$USER .`

### Image trop volumineuse / disque plein
Nettoie les images et conteneurs inutilisés :
```bash
docker system prune -a       # ⚠️ Supprime TOUT ce qui n'est pas utilisé
```

### "ModuleNotFoundError" malgré que la lib soit dans le Dockerfile
Tu as oublié de rebuild après modification du Dockerfile :
```bash
docker compose build --no-cache
docker compose up dashboard
```

---

## 💡 Astuces pro

### Faire un alias pour ne plus retaper les commandes

Dans `~/.bashrc` ou `~/.zshrc` (Linux/Mac) :
```bash
alias dc='docker compose'
alias dcu='docker compose up'
alias dcb='docker compose build'
alias dcr='docker compose run --rm'
```

Puis :
```bash
dcu dashboard            # au lieu de docker compose up dashboard
dcr test                 # au lieu de docker compose run --rm test
```

### Avoir un terminal Docker dans VS Code sans Dev Container

Si tu préfères garder VS Code sur ton hôte mais avoir un terminal Docker :
```bash
docker compose run --rm shell
```

### Voir l'usage des ressources
```bash
docker stats              # CPU, RAM, réseau en direct
```

---

## 🎓 Pour la soutenance

Mentionner Docker dans ta présentation montre la **maturité technique** :

> "Pour garantir la reproductibilité et éviter les problèmes de dépendances
> entre les machines de l'équipe, j'ai conteneurisé le dashboard avec Docker.
> N'importe qui peut cloner le projet et le lancer avec `docker compose up`
> sans rien installer d'autre que Docker — ça fonctionne identiquement
> sur Windows, Mac et Linux."

C'est un argument solide pour distinguer ton travail.

---

## 📚 Pour aller plus loin

- Docs Docker : <https://docs.docker.com/>
- Docs Dev Containers : <https://containers.dev/>
- Best practices Dockerfile : <https://docs.docker.com/develop/develop-images/dockerfile_best-practices/>
