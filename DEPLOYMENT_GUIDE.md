# Deployment Guide: Wildfire Risk Intelligence & Hotspot Analytics

This guide covers all methods to deploy the **Wildfire Risk Intelligence & Hotspot Analytics** platform so it is publicly accessible to evaluators, faculty, peers, and anyone around the world.

---

## 🌟 Method 1: Instant 1-Click Cloud Deployment on Render (Recommended Free 24/7 Hosting)

Render provides free, permanent 24/7 cloud hosting for full-stack Node.js web applications with automatic HTTPS.

### Steps:
1. **Push your code to GitHub:**
   Open a terminal in this project directory and run:
   ```bash
   # Create a new repository on https://github.com/new (e.g. named "wildfire-risk-intelligence")
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/wildfire-risk-intelligence.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Go to [dashboard.render.com](https://dashboard.render.com/) and sign in with GitHub.
   - Click **"New +"** $\to$ **"Web Service"**.
   - Select your `wildfire-risk-intelligence` repository.
   - Settings are automatically pre-configured by `render.yaml`:
     - **Name:** `wildfire-risk-intelligence`
     - **Environment:** `Node`
     - **Build Command:** *(leave blank or `npm install`)*
     - **Start Command:** `node web/server.js`
     - **Plan:** `Free`
   - Click **"Create Web Service"**.

3. **Your Live URL:**
   Within 60 seconds, Render will generate a live public URL:
   `https://wildfire-risk-intelligence.onrender.com`

---

## ⚡ Method 2: Instant Serverless Deployment on Vercel

Vercel provides blazing-fast edge deployment with a custom `*.vercel.app` domain.

### Steps:
1. Make sure you have pushed to GitHub (or install Vercel CLI).
2. Go to [vercel.com/new](https://vercel.com/new) and import your `wildfire-risk-intelligence` repository.
3. Vercel will automatically detect `vercel.json` and deploy `web/server.js`.
4. Click **Deploy**.
5. Your project will be live at:
   `https://wildfire-risk-intelligence.vercel.app`

*(Alternatively, run `npx vercel` in the project root from your command prompt).*

---

## 🌐 Method 3: 100% Free Static Deployment on GitHub Pages

The web application contains a built-in client-side simulation engine. It can run as a 100% standalone web app directly on **GitHub Pages** with zero backend infrastructure!

### Steps:
1. In your GitHub repository settings, go to **Pages** (under Code and automation).
2. Under **Build and deployment**:
   - Source: **Deploy from a branch**
   - Branch: `main`
   - Folder: `/web/public` (or create a `gh-pages` branch pointing to `web/public`)
3. Click **Save**.
4. Your site is live at:
   `https://<YOUR_GITHUB_USERNAME>.github.io/wildfire-risk-intelligence/`

---

## 🚀 Method 4: Instant Live Tunnel for Evaluation / Live Demos (Right Now)

If you are in an evaluation, presentation, or viva and need an immediate public link right now:

### Using Cloudflare Quick Tunnel (Zero Account Required):
Run:
```bash
.\cloudflared.exe tunnel --url http://localhost:3000
```
Cloudflare will immediately output a live, secure HTTPS link like:
`https://random-words.trycloudflare.com`  
Anyone on any mobile phone or external network can open that URL and access your live dashboard instantly!

---

## 🛡️ Architecture & Endpoints

When deployed, the platform serves:
- `GET /` — Responsive Leaflet.js Geovisualization & Analytical Dashboard
- `GET /api/analytics` — Precomputed multi-year metrics, 5-Fold Spatial CV results, and feature importances
- `GET /api/hotspots/live` — Near Real-Time 24-hour active fire detections from NASA FIRMS VIIRS 375m
- `POST /api/predict` — Dynamic meteorological risk calculator and incident command SOPs
