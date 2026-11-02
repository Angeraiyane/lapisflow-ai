# Démo enregistrée de 60 secondes (OBS Studio)
*Avant d'enregistrer : aucun secret à l'écran, repli Gemini désactivé (aucune variable `GEMINI_*` ni `FALLBACK_ENABLED`), terminal en grande police.*

## Préparation (hors enregistrement)
Terminal 1 : `python -m app.main`   (laisser tourner)
Terminal 2 : dossier du projet, venv actif.
Navigateur : onglet vide ; README ouvert dans un second onglet (tableau des résultats visible).

## Plan minuté
| Temps | Écran | Commande (PowerShell, terminal 2) |
|---|---|---|
| 0:00-0:08 | Il tourne | `Invoke-RestMethod http://127.0.0.1:8000/health` |
| 0:08-0:22 | Message courant → **auto** | `Invoke-RestMethod -Uri "http://127.0.0.1:8000/triage" -Method Post -ContentType "application/json" -Body '{"text": "I want to activate my new card please."}' \| ConvertTo-Json` |
| 0:22-0:40 | Message de fraude → **human**, avec la raison | `Invoke-RestMethod -Uri "http://127.0.0.1:8000/triage" -Method Post -ContentType "application/json" -Body '{"text": "Someone stole my card and took money from an ATM."}' \| ConvertTo-Json` |
| 0:40-0:50 | Ligne d'audit : hash seulement, pas de texte brut | `python scripts\show_audit.py` |
| 0:50-0:55 | Compteurs | `Invoke-RestMethod http://127.0.0.1:8000/stats` |
| 0:55-1:00 | 5 secondes sur le tableau de résultats du README | (navigateur) |

Dis une phrase par étape, honnête : « le routeur envoie 58 % des messages en automatique ; les intentions de fraude vont toujours à un humain ; mes résultats incluent un échec que je documente ».

## Tunnel public temporaire (test séparé, **non vérifié par moi**)
```
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://127.0.0.1:8000
```
Il affiche une adresse `https://….trycloudflare.com` (change à chaque lancement ; s'arrête quand tu fermes le terminal). Ouvre `/health` sur ton téléphone en données mobiles. Précautions : aucune authentification sur l'API, repli désactivé, données publiques seulement, tunnel fermé après le test. Pour le rejeu du 4 nov. : `python scripts\evaluate_live.py --base-url https://….trycloudflare.com --limit 300` (le rejeu complet de 1 962 messages à travers un tunnel peut être long).