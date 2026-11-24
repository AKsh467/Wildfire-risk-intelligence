/**
 * Node.js Server for Wildfire Risk Intelligence and Hotspot Analytics.
 *
 * Serves the interactive web interface, provides REST APIs for precomputed
 * spatial-temporal analytics, and evaluates real-time wildfire risk predictions.
 */

const http = require("http");
const fs = require("fs");
const path = require("path");
const url = require("url");

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, "public");
const ANALYTICS_FILE = path.join(__dirname, "..", "data", "processed", "analytics_summary.json");
const NRT_FILE = path.join(__dirname, "..", "data", "raw", "firms_nrt", "live_24h.json");

// MIME types for static files
const MIME_TYPES = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "application/javascript",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

/**
 * Calculate Tetens Vapor Pressure Deficit (VPD in kPa).
 */
function computeVpd(tempC, rhPercent) {
  const rhClamped = Math.max(0, Math.min(100, rhPercent));
  const es = 0.61078 * Math.exp((17.27 * tempC) / (tempC + 237.3));
  return Math.max(0, es * (1 - rhClamped / 100));
}

/**
 * Calculate Ångström Fire Hazard Index.
 */
function computeAngstrom(tempC, rhPercent) {
  return rhPercent / 20.0 + (27.0 - tempC) / 10.0;
}

/**
 * Map probability to operational risk tier.
 */
function assignRiskTier(prob) {
  if (prob < 0.2) return "Low";
  if (prob < 0.5) return "Moderate";
  if (prob < 0.75) return "High";
  return "Extreme";
}

/**
 * Evaluate relative risk for user-supplied weather inputs.
 */
function predictRisk(payload) {
  const temp = parseFloat(payload.T2M || 35.0);
  const rh = parseFloat(payload.RH2M || 25.0);
  const wind = parseFloat(payload.WS10M || 5.5);
  const rain7d = parseFloat(payload.rain_7d || 0.0);
  const dryDays = parseInt(payload.dry_days || 15);

  const vpd = computeVpd(temp, rh);
  const angstrom = computeAngstrom(temp, rh);

  // Atmospheric drying and fuel deficit weighting
  const vpdFactor = Math.min(1.0, vpd / 3.8);
  const drySpellFactor = Math.min(1.0, dryDays / 25.0);
  const windFactor = Math.min(1.0, wind / 15.0);
  const rainSuppression = Math.max(0.0, 1.0 - rain7d / 35.0);
  const hazardFactor = angstrom < 2.0 ? 1.0 : 0.45;

  let score =
    (0.42 * vpdFactor +
      0.26 * drySpellFactor +
      0.16 * windFactor +
      0.16 * hazardFactor) *
    rainSuppression;

  score = Math.max(0.02, Math.min(0.98, score));
  const tier = assignRiskTier(score);

  let guidance = "";
  if (tier === "Extreme") {
    guidance =
      "Critical evaporative demand (VPD > 3.0 kPa) with depleted surface fuel moisture. Ignitions can spread rapidly into crown fires. Suspend open agricultural burnings and stage rapid response crews.";
  } else if (tier === "High") {
    guidance =
      "Elevated fuel desiccation and brisk surface winds. Anomalies may transition into active fire fronts. Restrict biomass clearances and issue district watch advisories.";
  } else if (tier === "Moderate") {
    guidance =
      "Conditions permit ignition under continuous heat source, but fuel moisture buffers limit explosive propagation. Routine vigilance recommended.";
  } else {
    guidance =
      "Adequate surface moisture or recent rainfall suppressing ignition potential. Standard baseline monitoring.";
  }

  return {
    risk_score: parseFloat(score.toFixed(4)),
    risk_tier: tier,
    vpd_kpa: parseFloat(vpd.toFixed(3)),
    angstrom_index: parseFloat(angstrom.toFixed(2)),
    guidance: guidance,
    inputs: { temp, rh, wind, rain7d, dryDays },
  };
}

const server = http.createServer((req, res) => {
  const parsedUrl = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const pathname = parsedUrl.pathname;

  // CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  // REST API: GET /api/analytics
  if (req.method === "GET" && pathname === "/api/analytics") {
    if (fs.existsSync(ANALYTICS_FILE)) {
      const data = fs.readFileSync(ANALYTICS_FILE, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(data);
    } else {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Analytics summary file not found. Please run the analysis pipeline first." }));
    }
    return;
  }

  // REST API: GET /api/hotspots/live
  if (req.method === "GET" && pathname === "/api/hotspots/live") {
    if (fs.existsSync(NRT_FILE)) {
      const data = fs.readFileSync(NRT_FILE, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(data);
    } else {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Live NRT satellite feed not found." }));
    }
    return;
  }

  // REST API: POST /api/hotspots/refresh
  if (req.method === "POST" && pathname === "/api/hotspots/refresh") {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      try {
        const payload = JSON.parse(body || "{}");
        const mapKey = (payload.map_key || "").trim();
        const pythonExe = path.join(__dirname, "..", ".venv", "Scripts", "python.exe");
        const scriptPath = path.join(__dirname, "..", "src", "ingestion", "firms_nrt_client.py");
        const { spawn } = require("child_process");
        const args = [scriptPath, "--force"];
        if (mapKey) {
          args.push("--map-key", mapKey);
        }
        const child = spawn(fs.existsSync(pythonExe) ? pythonExe : "python", args, {
          cwd: path.join(__dirname, ".."),
          env: { ...process.env, PYTHONPATH: path.join(__dirname, "..") },
        });

        child.on("close", (code) => {
          if (fs.existsSync(NRT_FILE)) {
            const data = fs.readFileSync(NRT_FILE, "utf-8");
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(data);
          } else {
            res.writeHead(500, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "Failed to update NRT feed", exit_code: code }));
          }
        });
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Invalid JSON", details: err.message }));
      }
    });
    return;
  }

  // REST API: POST /api/predict
  if (req.method === "POST" && pathname === "/api/predict") {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      try {
        const payload = JSON.parse(body || "{}");
        const result = predictRisk(payload);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Invalid JSON payload", details: err.message }));
      }
    });
    return;
  }

  // Static File Serving
  let filePath = path.join(PUBLIC_DIR, pathname === "/" ? "index.html" : pathname);
  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || "text/plain";

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === "ENOENT") {
        // Fallback to index.html for single-page routing
        fs.readFile(path.join(PUBLIC_DIR, "index.html"), (err2, content2) => {
          if (err2) {
            res.writeHead(404, { "Content-Type": "text/plain" });
            res.end("404 Not Found");
          } else {
            res.writeHead(200, { "Content-Type": "text/html" });
            res.end(content2, "utf-8");
          }
        });
      } else {
        res.writeHead(500);
        res.end(`Server Error: ${err.code}`);
      }
    } else {
      res.writeHead(200, { "Content-Type": contentType });
      res.end(content, "utf-8");
    }
  });
});

server.listen(PORT, () => {
  console.log("==================================================================");
  console.log(`🔥 Wildfire Risk Intelligence - Node.js Web Server`);
  console.log(`📡 URL: http://localhost:${PORT}`);
  console.log(`📊 API Endpoints:`);
  console.log(`   - GET  /api/analytics`);
  console.log(`   - GET  /api/hotspots/live`);
  console.log(`   - POST /api/predict`);
  console.log("==================================================================");
});
