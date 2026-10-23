# NeuralFetch Web Frontend

A responsive web application for NeuralFetch built with **HTML5, Tailwind CSS, JavaScript, Lucide Icons, Marked.js, and Canvas Confetti**. It visualizes the LangGraph 4-node multi-agent pipeline in real-time.

---

## 🚀 Easy Deployment

The `frontend/` directory is completely static, with zero build steps or node_modules required.

### Deploy on Netlify (10 Seconds)
1. **Drag & Drop**:
   - Log into [app.netlify.com](https://app.netlify.com).
   - Drag and drop the `frontend/` folder into the Netlify dashboard.
2. **Or via Netlify CLI**:
   ```bash
   cd frontend
   netlify deploy --prod
   ```
3. Open your deployed Netlify URL. Click the **Settings (gear icon)** at the top right to configure your backend API URL if running on a remote host or ngrok tunnel.

---

### Deploy on Vercel (10 Seconds)
1. **Via Vercel CLI**:
   ```bash
   cd frontend
   vercel --prod
   ```
2. **Or via Vercel Dashboard**:
   - Push to GitHub and import the repository into [vercel.com](https://vercel.com).
   - Set **Root Directory** to `frontend`.
   - Click **Deploy**.

---

## 💻 Local Testing & Development

### Option A: Served directly by FastAPI (Unified Local Port)
Start your NeuralFetch backend server:
```bash
uvicorn app.api.server:app --reload
```
Open **`http://localhost:8000`** in any browser. The FastAPI server automatically mounts and serves the frontend at the root URL while keeping all API endpoints under `/api/v1/`.

### Option B: Using any static HTTP server
```bash
# Python:
python -m http.server 3000 --directory frontend

# Or Node:
npx serve frontend
```
Then visit `http://localhost:3000`.
