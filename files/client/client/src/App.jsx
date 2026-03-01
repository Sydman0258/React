import { useState, useEffect, useCallback } from "react";
import "./App.css";

const API = "http://localhost:3001/api";

// ── helpers ─────────────────────────────────────
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const badge = (p) =>
  p === "H" ? "home-win" : p === "A" ? "away-win" : "draw";
const badgeLabel = (p) =>
  p === "H" ? "Home Win" : p === "A" ? "Away Win" : "Draw";

function ConfidenceBar({ home, draw, away }) {
  return (
    <div className="conf-bar">
      <div className="conf-seg home" style={{ width: pct(home) }} title={`Home ${pct(home)}`}>
        {home > 0.12 && pct(home)}
      </div>
      <div className="conf-seg draw" style={{ width: pct(draw) }} title={`Draw ${pct(draw)}`}>
        {draw > 0.12 && pct(draw)}
      </div>
      <div className="conf-seg away" style={{ width: pct(away) }} title={`Away ${pct(away)}`}>
        {away > 0.12 && pct(away)}
      </div>
    </div>
  );
}

function MatchCard({ match, index }) {
  const { home_team, away_team, date, prediction, prediction_label, confidence, probabilities } = match;
  return (
    <div className={`match-card animate-in`} style={{ animationDelay: `${index * 60}ms` }}>
      <div className="match-date">{date}</div>
      <div className="match-teams">
        <span className="team home-team">{home_team}</span>
        <span className="vs-badge">VS</span>
        <span className="team away-team">{away_team}</span>
      </div>
      <ConfidenceBar home={probabilities.home_win} draw={probabilities.draw} away={probabilities.away_win} />
      <div className="match-footer">
        <span className={`result-tag ${badge(prediction)}`}>{prediction_label}</span>
        <span className="confidence-text">{pct(confidence)} confidence</span>
      </div>
    </div>
  );
}

function PredictTab({ teams }) {
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const predict = async () => {
    if (!home || !away || home === away) {
      setError("Pick two different teams.");
      return;
    }
    setLoading(true); setError(""); setResult(null);
    try {
      const r = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ home_team: home, away_team: away }),
      });
      const d = await r.json();
      if (d.error) setError(d.error);
      else setResult(d);
    } catch {
      setError("Could not reach API. Make sure Express + Python are running.");
    }
    setLoading(false);
  };

  return (
    <div className="tab-content">
      <div className="predict-form">
        <h2 className="section-title">Single Match Predictor</h2>
        <div className="team-selectors">
          <div className="selector-group">
            <label>Home Team</label>
            <select value={home} onChange={(e) => setHome(e.target.value)}>
              <option value="">-- Select --</option>
              {teams.map((t) => (
                <option key={t.name} value={t.name}>{t.name}</option>
              ))}
            </select>
          </div>
          <div className="vs-center">
            <div className="vs-circle">VS</div>
          </div>
          <div className="selector-group">
            <label>Away Team</label>
            <select value={away} onChange={(e) => setAway(e.target.value)}>
              <option value="">-- Select --</option>
              {teams.filter((t) => t.name !== home).map((t) => (
                <option key={t.name} value={t.name}>{t.name}</option>
              ))}
            </select>
          </div>
        </div>
        <button className="predict-btn" onClick={predict} disabled={loading || !home || !away}>
          {loading ? <span className="spinner" /> : "⚽ Predict Match"}
        </button>
        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && (
        <div className="result-panel animate-in">
          <div className="result-header">
            <span className="result-home">{result.home_team}</span>
            <div className="result-center">
              <div className={`big-result-tag ${badge(result.prediction)}`}>{result.prediction_label}</div>
              <div className="expected-goals">
                Expected: {result.expected_goals.home} – {result.expected_goals.away}
              </div>
            </div>
            <span className="result-away">{result.away_team}</span>
          </div>
          <ConfidenceBar
            home={result.probabilities.home_win}
            draw={result.probabilities.draw}
            away={result.probabilities.away_win}
          />
          <div className="prob-labels">
            <span>🏠 {pct(result.probabilities.home_win)}</span>
            <span>🤝 {pct(result.probabilities.draw)}</span>
            <span>✈️ {pct(result.probabilities.away_win)}</span>
          </div>
          <div className="confidence-badge">
            Model confidence: <strong>{pct(result.confidence)}</strong>
          </div>
        </div>
      )}
    </div>
  );
}

function FixturesTab() {
  const [fixtures, setFixtures] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetched, setFetched] = useState(false);
  const [error, setError] = useState("");

  const loadAndPredict = async () => {
    setLoading(true); setError("");
    try {
      const fr = await fetch(`${API}/fixtures/upcoming`);
      const fd = await fr.json();
      setFixtures(fd.fixtures);

      const pr = await fetch(`${API}/predict/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fixtures: fd.fixtures }),
      });
      const pd = await pr.json();
      if (pd.error) setError(pd.error);
      else { setPredictions(pd.predictions); setFetched(true); }
    } catch {
      setError("Could not reach API.");
    }
    setLoading(false);
  };

  return (
    <div className="tab-content">
      <div className="fixtures-header">
        <h2 className="section-title">Upcoming Gameweek</h2>
        <button className="predict-btn sm" onClick={loadAndPredict} disabled={loading}>
          {loading ? <span className="spinner" /> : "Load & Predict All"}
        </button>
      </div>
      {error && <div className="error-msg">{error}</div>}
      {!fetched && !loading && (
        <div className="empty-state">Click "Load & Predict All" to fetch fixtures and run predictions.</div>
      )}
      <div className="matches-grid">
        {predictions.map((m, i) => (
          <MatchCard key={i} match={m} index={i} />
        ))}
      </div>
    </div>
  );
}

function ModelTab({ stats, onTrain }) {
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState(null);

  const handleTrain = async () => {
    setTraining(true);
    const r = await onTrain();
    setTrainResult(r);
    setTraining(false);
  };

  const importances = stats?.feature_importances
    ? Object.entries(stats.feature_importances).sort((a, b) => b[1] - a[1])
    : [];

  const maxImp = importances[0]?.[1] || 1;

  return (
    <div className="tab-content">
      <h2 className="section-title">Model Dashboard</h2>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats ? pct(stats.accuracy) : "—"}</div>
          <div className="stat-label">Test Accuracy</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.teams_count ?? "—"}</div>
          <div className="stat-label">Teams Tracked</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.training_samples?.toLocaleString() ?? "—"}</div>
          <div className="stat-label">Training Matches</div>
        </div>
        <div className="stat-card highlight">
          <div className="stat-value">RF 200</div>
          <div className="stat-label">Random Forest Trees</div>
        </div>
      </div>

      <button className="predict-btn train-btn" onClick={handleTrain} disabled={training}>
        {training ? <><span className="spinner" /> Training…</> : "🔁 Retrain Model"}
      </button>

      {trainResult && (
        <div className="train-result animate-in">
          ✅ Training complete! Accuracy: <strong>{trainResult.accuracy_pct}</strong>
        </div>
      )}

      {importances.length > 0 && (
        <div className="importance-section">
          <h3>Feature Importances</h3>
          {importances.map(([feat, val]) => (
            <div key={feat} className="imp-row">
              <span className="feat-name">{feat.replace(/_/g, " ")}</span>
              <div className="imp-bar-bg">
                <div
                  className="imp-bar-fill"
                  style={{ width: `${(val / maxImp) * 100}%` }}
                />
              </div>
              <span className="feat-val">{(val * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StandingsTab({ teams }) {
  return (
    <div className="tab-content">
      <h2 className="section-title">Team Strength Rankings</h2>
      <div className="standings-table">
        <div className="st-header">
          <span>#</span><span>Team</span><span>Strength</span><span>Rating</span>
        </div>
        {teams.map((t, i) => (
          <div key={t.name} className={`st-row animate-in`} style={{ animationDelay: `${i * 30}ms` }}>
            <span className="st-pos">{i + 1}</span>
            <span className="st-name">{t.name}</span>
            <div className="strength-bar-bg">
              <div className="strength-bar-fill" style={{ width: `${t.strength * 100}%` }} />
            </div>
            <span className="st-score">{(t.strength * 100).toFixed(0)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main App ────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("predict");
  const [teams, setTeams] = useState([]);
  const [stats, setStats] = useState(null);
  const [mlStatus, setMlStatus] = useState("checking");

  const loadData = useCallback(async () => {
    try {
      const [tr, sr] = await Promise.all([
        fetch(`${API}/teams`),
        fetch(`${API}/stats`),
      ]);
      const td = await tr.json();
      const sd = await sr.json();
      setTeams(td.teams || []);
      setStats(sd);
      setMlStatus(sd.model_trained ? "ready" : "untrained");
    } catch {
      setMlStatus("offline");
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const handleTrain = async () => {
    try {
      const r = await fetch(`${API}/train`, { method: "POST" });
      const d = await r.json();
      await loadData();
      return d;
    } catch {
      return { error: true };
    }
  };

  const statusInfo = {
    checking: { cls: "status-checking", text: "Checking…" },
    ready: { cls: "status-ready", text: "Model Ready" },
    untrained: { cls: "status-warn", text: "Model Not Trained" },
    offline: { cls: "status-error", text: "Services Offline" },
  }[mlStatus];

  const TABS = [
    { id: "predict", label: "⚽ Predict" },
    { id: "fixtures", label: "📅 Fixtures" },
    { id: "standings", label: "🏆 Teams" },
    { id: "model", label: "🤖 Model" },
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⚽</span>
            <div>
              <div className="logo-title">FootballIQ</div>
              <div className="logo-sub">AI Match Predictor</div>
            </div>
          </div>
          <div className={`ml-status ${statusInfo.cls}`}>
            <span className="status-dot" />
            {statusInfo.text}
            {mlStatus === "untrained" && (
              <button className="mini-train-btn" onClick={handleTrain}>Train Now</button>
            )}
          </div>
        </div>
        <nav className="nav">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`nav-btn ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="main">
        {tab === "predict" && <PredictTab teams={teams} />}
        {tab === "fixtures" && <FixturesTab />}
        {tab === "standings" && <StandingsTab teams={teams} />}
        {tab === "model" && <ModelTab stats={stats} onTrain={handleTrain} />}
      </main>

      <footer className="footer">
        FootballIQ · Powered by Random Forest + Flask + Express ·
      </footer>
    </div>
  );
}
