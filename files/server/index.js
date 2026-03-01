const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = 3001;
const ML_URL = process.env.ML_URL || "http://localhost:5001";

app.use(cors());
app.use(express.json());

// ── Proxy helpers ──────────────────────────────
async function mlGet(path) {
  const res = await axios.get(`${ML_URL}${path}`);
  return res.data;
}
async function mlPost(path, body) {
  const res = await axios.post(`${ML_URL}${path}`, body);
  return res.data;
}

// ── Routes ─────────────────────────────────────

/** Health check */
app.get("/api/health", async (req, res) => {
  try {
    const ml = await mlGet("/health");
    res.json({ api: "ok", ml });
  } catch {
    res.status(503).json({ api: "ok", ml: "unreachable" });
  }
});

/** Get all available teams */
app.get("/api/teams", async (req, res) => {
  try {
    const data = await mlGet("/teams");
    res.json(data);
  } catch (e) {
    res.status(502).json({ error: "ML service unavailable", detail: e.message });
  }
});

/** Train the model */
app.post("/api/train", async (req, res) => {
  try {
    const data = await mlPost("/train", {});
    res.json(data);
  } catch (e) {
    res.status(502).json({ error: "Training failed", detail: e.message });
  }
});

/** Single match prediction */
app.post("/api/predict", async (req, res) => {
  const { home_team, away_team, h2h_home_wins, h2h_draws } = req.body;
  if (!home_team || !away_team) {
    return res.status(400).json({ error: "home_team and away_team are required" });
  }
  if (home_team === away_team) {
    return res.status(400).json({ error: "Teams must be different" });
  }
  try {
    const data = await mlPost("/predict", { home_team, away_team, h2h_home_wins, h2h_draws });
    res.json(data);
  } catch (e) {
    res.status(502).json({ error: "Prediction failed", detail: e.message });
  }
});

/** Batch predict upcoming fixtures */
app.post("/api/predict/batch", async (req, res) => {
  const { fixtures } = req.body;
  if (!fixtures || !Array.isArray(fixtures)) {
    return res.status(400).json({ error: "fixtures array required" });
  }
  try {
    const data = await mlPost("/batch-predict", { fixtures });
    res.json(data);
  } catch (e) {
    res.status(502).json({ error: "Batch prediction failed", detail: e.message });
  }
});

/** Model stats */
app.get("/api/stats", async (req, res) => {
  try {
    const data = await mlGet("/stats");
    res.json(data);
  } catch (e) {
    res.status(502).json({ error: "ML service unavailable", detail: e.message });
  }
});

/** Generate a sample upcoming gameweek */
app.get("/api/fixtures/upcoming", async (req, res) => {
  const fixtures = [
    { home_team: "Manchester City", away_team: "Arsenal", date: nextMatchday(0) },
    { home_team: "Liverpool", away_team: "Chelsea", date: nextMatchday(0) },
    { home_team: "Manchester United", away_team: "Tottenham", date: nextMatchday(0) },
    { home_team: "Newcastle", away_team: "Aston Villa", date: nextMatchday(1) },
    { home_team: "Brighton", away_team: "West Ham", date: nextMatchday(1) },
    { home_team: "Brentford", away_team: "Crystal Palace", date: nextMatchday(1) },
    { home_team: "Wolves", away_team: "Everton", date: nextMatchday(2) },
    { home_team: "Fulham", away_team: "Bournemouth", date: nextMatchday(2) },
    { home_team: "Nottingham Forest", away_team: "Burnley", date: nextMatchday(2) },
    { home_team: "Luton Town", away_team: "Sheffield United", date: nextMatchday(2) },
  ];
  res.json({ fixtures });
});

function nextMatchday(offsetDays) {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays + (6 - d.getDay() + 7) % 7 || 7);
  return d.toISOString().split("T")[0];
}

app.listen(PORT, () => {
  console.log(`⚡ Express API running on http://localhost:${PORT}`);
  console.log(`🔗 Proxying ML service at ${ML_URL}`);
});
