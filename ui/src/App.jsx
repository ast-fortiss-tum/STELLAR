import { useEffect, useMemo, useState } from "react";
import { PRELOADED_CONFIGS } from "./preloadedConfigs";

const defaultFeatures = PRELOADED_CONFIGS[0].content;

const initialForm = {
  featuresContent: defaultFeatures,
  turnMode: "single",
  testingTime: 120,
  populationSize: 40,
  judgeModel: "gpt-4.1-mini",
  judgeTemperature: 0.2,
  fitnessResponseEnabled: true,
  fitnessResponseThreshold: 0.75,
  fitnessContentEnabled: true,
  fitnessContentThreshold: 0.75,
  fitnessDiverseEnabled: true,
  fitnessEfficiencyEnabled: true,
  fitnessEfficiencyThreshold: 0.75,
  sut: "IPA_LOS",
  enableWandb: false,
  wandbEntity: "opentest",
  wandbProject: "demo",
  wandbLimit: 20,
  endpoint: "http://localhost:8000/experiments/start"
};

const parseFeatureConfig = (content) => {
  try {
    const parsed = JSON.parse(content);
    const categorical = Array.isArray(parsed.categorical_features) ? parsed.categorical_features : [];
    const ordinal = Array.isArray(parsed.ordinal_features) ? parsed.ordinal_features : [];

    const features = [
      ...categorical.map((feature) => ({ ...feature, type: "categorical" })),
      ...ordinal.map((feature) => ({ ...feature, type: "ordinal" }))
    ].filter((feature) => feature && typeof feature.name === "string");

    return {
      ok: true,
      features,
      error: null
    };
  } catch (error) {
    return {
      ok: false,
      features: [],
      error: error.message
    };
  }
};

const formatFeatureValue = (value) => {
  if (value === null) {
    return "null";
  }

  return String(value);
};

const STORAGE_KEY_RUNS = "stellar_ui_runs";

const loadRunHistory = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_RUNS);
    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

const withLogLine = (message) => ({
  at: new Date().toISOString(),
  message
});

const parseEnvText = (raw) => {
  const env = {};

  raw.split("\n").forEach((lineRaw) => {
    const line = lineRaw.trim();
    if (!line || line.startsWith("#")) {
      return;
    }

    const cleaned = line.startsWith("export ") ? line.slice(7).trim() : line;
    const index = cleaned.indexOf("=");
    if (index < 0) {
      return;
    }

    const key = cleaned.slice(0, index).trim();
    let value = cleaned.slice(index + 1).trim();
    if (!key) {
      return;
    }

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    env[key] = value;
  });

  return env;
};

const stringifyEnv = (envObj) =>
  Object.entries(envObj)
    .map(([key, value]) => `${key}=${value}`)
    .join("\n");

const buildRunStatusUrl = (startEndpoint, backendRunId) => {
  if (!startEndpoint || !backendRunId) {
    return null;
  }

  try {
    const url = new URL(startEndpoint);
    url.pathname = `/experiments/runs/${backendRunId}`;
    url.search = "";
    return url.toString();
  } catch {
    return null;
  }
};

const buildRunStopUrl = (startEndpoint, backendRunId) => {
  if (!startEndpoint || !backendRunId) {
    return null;
  }

  try {
    const url = new URL(startEndpoint);
    url.pathname = `/experiments/runs/${backendRunId}/stop`;
    url.search = "";
    return url.toString();
  } catch {
    return null;
  }
};

const buildEnvDefaultsUrls = (startEndpoint) => {
  if (!startEndpoint) {
    return [];
  }

  try {
    const url = new URL(startEndpoint);
    const urls = [];

    const primary = new URL(url.toString());
    primary.pathname = "/experiments/env-defaults";
    primary.search = "";
    urls.push(primary.toString());

    const legacy = new URL(url.toString());
    legacy.pathname = "/env-defaults";
    legacy.search = "";
    urls.push(legacy.toString());

    return urls;
  } catch {
    return [];
  }
};

const buildWandbRunsUrl = (startEndpoint, entity, project, limit) => {
  if (!startEndpoint) {
    return null;
  }

  try {
    const url = new URL(startEndpoint);
    url.pathname = "/wandb/runs";
    url.search = "";
    if (entity) {
      url.searchParams.set("entity", entity);
    }
    if (project) {
      url.searchParams.set("project", project);
    }
    if (limit) {
      url.searchParams.set("limit", String(limit));
    }
    return url.toString();
  } catch {
    return null;
  }
};

function App() {
  const [form, setForm] = useState(initialForm);
  const [preloadKey, setPreloadKey] = useState(PRELOADED_CONFIGS[0].key);
  const [domainView, setDomainView] = useState("visual");
  const [featureSelections, setFeatureSelections] = useState({});
  const [activeTab, setActiveTab] = useState("config");
  const [runHistory, setRunHistory] = useState(() => loadRunHistory());
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Ready to configure and launch experiments.");
  const [envText, setEnvText] = useState("");
  const [envStatus, setEnvStatus] = useState("idle");
  const [stoppingRuns, setStoppingRuns] = useState({});
  const [wandbStatus, setWandbStatus] = useState("Not loaded");
  const [wandbRuns, setWandbRuns] = useState([]);

  const parsedFeatureConfig = useMemo(() => parseFeatureConfig(form.featuresContent), [form.featuresContent]);

  useEffect(() => {
    if (!parsedFeatureConfig.ok) {
      return;
    }

    setFeatureSelections((prev) => {
      const next = {};

      parsedFeatureConfig.features.forEach((feature) => {
        if (Object.prototype.hasOwnProperty.call(prev, feature.name)) {
          next[feature.name] = prev[feature.name];
        } else {
          next[feature.name] = feature.values && feature.values.length ? 0 : -1;
        }
      });

      const nextKeys = Object.keys(next);
      const prevKeys = Object.keys(prev);
      const unchanged =
        nextKeys.length === prevKeys.length && nextKeys.every((key) => prev[key] === next[key]);

      return unchanged ? prev : next;
    });
  }, [parsedFeatureConfig]);

  const canStart = useMemo(() => {
    return (
      form.featuresContent.trim().length > 0 &&
      parsedFeatureConfig.ok &&
      Number(form.testingTime) > 0 &&
      Number(form.populationSize) > 0 &&
      Number(form.judgeTemperature) >= 0 &&
      Number(form.judgeTemperature) <= 2 &&
      Number(form.fitnessResponseThreshold) >= 0 &&
      Number(form.fitnessResponseThreshold) <= 1 &&
      Number(form.fitnessContentThreshold) >= 0 &&
      Number(form.fitnessContentThreshold) <= 1 &&
      Number(form.fitnessEfficiencyThreshold) >= 0 &&
      Number(form.fitnessEfficiencyThreshold) <= 1 &&
      form.endpoint.trim().length > 0
    );
  }, [form, parsedFeatureConfig.ok]);

  const updateField = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const addRunRecord = (record) => {
    setRunHistory((prev) => {
      const next = [record, ...prev];
      localStorage.setItem(STORAGE_KEY_RUNS, JSON.stringify(next));
      return next;
    });
  };

  const updateRunRecord = (runId, updateFn) => {
    setRunHistory((prev) => {
      const next = prev.map((run) => {
        if (run.id !== runId) {
          return run;
        }

        return updateFn(run);
      });

      localStorage.setItem(STORAGE_KEY_RUNS, JSON.stringify(next));
      return next;
    });
  };

  useEffect(() => {
    const activeRuns = runHistory.filter(
      (run) =>
        Boolean(run.backendRunId) &&
        ["queued", "running", "started"].includes(run.status)
    );

    if (activeRuns.length === 0) {
      return;
    }

    const timer = setInterval(async () => {
      await Promise.all(
        activeRuns.map(async (run) => {
          const statusUrl = buildRunStatusUrl(run.endpoint, run.backendRunId);

          if (!statusUrl) {
            return;
          }

          try {
            const response = await fetch(statusUrl);
            if (!response.ok) {
              return;
            }

            const data = await response.json();
            updateRunRecord(run.id, (prev) => ({
              ...prev,
              status: data.status || prev.status,
              note:
                data.returnCode === null || data.returnCode === undefined
                  ? prev.note
                  : `Process exit code: ${data.returnCode}`,
              logs: Array.isArray(data.logs) && data.logs.length > 0 ? data.logs : prev.logs
            }));
          } catch {
            // Keep previous state on transient poll failures.
          }
        })
      );
    }, 2000);

    return () => clearInterval(timer);
  }, [runHistory]);

  const loadEnvDefaults = async () => {
    const urls = buildEnvDefaultsUrls(form.endpoint);
    if (urls.length === 0) {
      setEnvStatus("Endpoint URL is invalid.");
      return;
    }

    setEnvStatus("Loading .env defaults...");

    try {
      let data = null;
      let lastStatus = "404";

      for (const url of urls) {
        const response = await fetch(url);
        if (response.ok) {
          data = await response.json();
          break;
        }
        lastStatus = String(response.status);
      }

      if (!data) {
        throw new Error(`Request failed with status ${lastStatus}`);
      }

      const defaults = data && typeof data === "object" ? data.env || {} : {};
      const source = data && typeof data === "object" ? data.source || "unknown source" : "unknown source";
      setEnvText(stringifyEnv(defaults));
      const count = Object.keys(defaults).length;
      if (count === 0) {
        setEnvStatus(`Loaded 0 variables from ${source}. File exists but has no usable KEY=VALUE pairs.`);
      } else {
        setEnvStatus(`Loaded ${count} variables from ${source}`);
      }
    } catch (error) {
      setEnvStatus(`Could not load .env defaults: ${error.message}`);
    }
  };

  const loadWandbRuns = async () => {
    const url = buildWandbRunsUrl(
      form.endpoint,
      form.wandbEntity,
      form.wandbProject,
      form.wandbLimit
    );

    if (!url) {
      setWandbStatus("Endpoint URL is invalid.");
      return;
    }

    setWandbStatus("Loading W&B runs...");
    try {
      const response = await fetch(url);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || `Request failed with status ${response.status}`);
      }

      const runs = Array.isArray(data.runs) ? data.runs : [];
      const entity = data.entity || form.wandbEntity || "opentest";
      const project = data.project || form.wandbProject || "demo";
      setWandbRuns(runs);
      const sourceText = data.envSource ? ` via ${data.envSource}` : "";
      setWandbStatus(`Loaded ${runs.length} runs from ${entity}/${project}${sourceText}`);

      if (!Array.isArray(data.runs)) {
        setWandbStatus(
          `Loaded ${runs.length} runs from ${entity}/${project}${sourceText}. Warning: backend response missing 'runs' array.`
        );
      }
    } catch (error) {
      setWandbStatus(`Could not load W&B runs: ${error.message}`);
    }
  };

  useEffect(() => {
    // Auto-load .env defaults when the endpoint is configured.
    if (!form.endpoint || envText.trim().length > 0) {
      return;
    }

    loadEnvDefaults();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.endpoint]);

  const handlePreloadChange = (key) => {
    setPreloadKey(key);
    const selected = PRELOADED_CONFIGS.find((config) => config.key === key);

    if (selected) {
      updateField("featuresContent", selected.content);
      setDomainView("visual");
      setStatus("idle");
      setMessage(`Loaded ${selected.label}.`);
    }
  };

  const handleConfigFileUpload = async (event) => {
    const [file] = event.target.files || [];

    if (!file) {
      return;
    }

    try {
      const content = await file.text();
      updateField("featuresContent", content);
      setPreloadKey("uploaded");
      setDomainView("visual");
      setStatus("idle");
      setMessage(`Loaded local file: ${file.name}`);
    } catch (error) {
      setStatus("error");
      setMessage(`Could not read file: ${error.message}`);
    }

    event.target.value = "";
  };

  const handleFeatureSelectionChange = (featureName, selectedIndex) => {
    const index = Number(selectedIndex);

    setFeatureSelections((prev) => ({
      ...prev,
      [featureName]: Number.isNaN(index) ? -1 : index
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!canStart) {
      setStatus("error");
      setMessage("Please fix invalid values before starting.");
      return;
    }

    const selectedFeatureValues = parsedFeatureConfig.features.reduce((acc, feature) => {
      const selectedIndex = featureSelections[feature.name];

      if (
        Number.isInteger(selectedIndex) &&
        selectedIndex >= 0 &&
        Array.isArray(feature.values) &&
        selectedIndex < feature.values.length
      ) {
        acc[feature.name] = feature.values[selectedIndex];
      } else {
        acc[feature.name] = null;
      }

      return acc;
    }, {});

    const payload = {
      domainConfig: {
        featuresContent: form.featuresContent,
        selectedFeatureValues
      },
      executionParameters: {
        sut: form.sut,
        turnMode: form.turnMode,
        testingTimeMinutes: Number(form.testingTime),
        populationSize: Number(form.populationSize),
        enableWandb: Boolean(form.enableWandb),
        envOverrides: parseEnvText(envText)
      },
      evaluationParameters: {
        fitnessFunctions: {
          response: {
            name: "answer_fitness",
            enabled: Boolean(form.fitnessResponseEnabled),
            judgeModel: form.judgeModel,
            temperature: Number(form.judgeTemperature),
            threshold: Number(form.fitnessResponseThreshold)
          },
          content: {
            name: "content_fitness",
            enabled: Boolean(form.fitnessContentEnabled),
            threshold: Number(form.fitnessContentThreshold)
          },
          diverse: {
            name: "distance",
            enabled: Boolean(form.fitnessDiverseEnabled)
          },
          efficiency: {
            name: "efficiency_fitness",
            enabled: Boolean(form.fitnessEfficiencyEnabled),
            threshold: Number(form.fitnessEfficiencyThreshold)
          }
        }
      },
      startedAt: new Date().toISOString()
    };

    const runId = `run-${Date.now()}`;

    addRunRecord({
      id: runId,
      startedAt: payload.startedAt,
      sut: form.sut,
      turnMode: form.turnMode,
      populationSize: Number(form.populationSize),
      testingTimeMinutes: Number(form.testingTime),
      launchMode: "bridge",
      status: "queued",
      backendRunId: null,
      endpoint: form.endpoint,
      note: null,
      logs: [
        withLogLine(`Run created for ${form.sut}`),
        withLogLine("Configuration payload prepared"),
        withLogLine(form.enableWandb ? "W&B logging enabled" : "W&B logging disabled (--no_wandb)"),
        withLogLine(`Env overrides: ${Object.keys(parseEnvText(envText)).length}`),
        withLogLine(`Dispatching POST request to ${form.endpoint}`)
      ]
    });

    setStatus("running");
    setMessage("Starting experiment...");
    setActiveTab("runs");

    try {
      const response = await fetch(form.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const result = await response.json().catch(() => ({}));
      setStatus("success");
      setMessage(result.message || "Experiment accepted by backend.");
      updateRunRecord(runId, (run) => ({
        ...run,
        status: result.status || "running",
        backendRunId: result.runId || null,
        note: result.message || null,
        logs: [
          ...(run.logs || []),
          withLogLine(result.runId ? `Backend run id: ${result.runId}` : "Backend accepted run"),
          withLogLine(result.message || "Backend acknowledged run start")
        ]
      }));
    } catch (error) {
      setStatus("error");
      setMessage(`Could not reach backend: ${error.message}`);
      updateRunRecord(runId, (run) => ({
        ...run,
        status: "failed",
        note: error.message,
        logs: [...(run.logs || []), withLogLine(`Error: ${error.message}`)]
      }));
    }
  };

  const handleStopRun = async (run) => {
    const stopUrl = buildRunStopUrl(run.endpoint, run.backendRunId);
    if (!stopUrl) {
      updateRunRecord(run.id, (prev) => ({
        ...prev,
        logs: [...(prev.logs || []), withLogLine("Stop failed: missing backend run id or endpoint")]
      }));
      return;
    }

    setStoppingRuns((prev) => ({ ...prev, [run.id]: true }));
    updateRunRecord(run.id, (prev) => ({
      ...prev,
      logs: [...(prev.logs || []), withLogLine("Sending stop request...")]
    }));

    try {
      const response = await fetch(stopUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        }
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || `Request failed with status ${response.status}`);
      }

      updateRunRecord(run.id, (prev) => ({
        ...prev,
        status: data.status || "stopped",
        note: data.message || prev.note,
        logs: [...(prev.logs || []), withLogLine(data.message || "Stop requested")]
      }));
    } catch (error) {
      updateRunRecord(run.id, (prev) => ({
        ...prev,
        logs: [...(prev.logs || []), withLogLine(`Stop failed: ${error.message}`)]
      }));
    } finally {
      setStoppingRuns((prev) => ({ ...prev, [run.id]: false }));
    }
  };

  return (
    <div className="page-shell">
      <div className="ambient-shape shape-a" />
      <div className="ambient-shape shape-b" />
      <main className="content">
        <header className="hero">
          <p className="kicker">STELLAR</p>
          <h1>Experiment Configuration Console</h1>
          <p className="subtitle">
            Configure domain features, tune execution behavior, define evaluation logic, then launch your run.
          </p>
        </header>

        <div className="tab-bar" role="tablist" aria-label="STELLAR views">
          <button
            className={`tab-btn ${activeTab === "config" ? "active" : ""}`}
            type="button"
            onClick={() => setActiveTab("config")}
          >
            Configure
          </button>
          <button
            className={`tab-btn ${activeTab === "runs" ? "active" : ""}`}
            type="button"
            onClick={() => setActiveTab("runs")}
          >
            Runs ({runHistory.length})
          </button>
          <button
            className={`tab-btn ${activeTab === "env" ? "active" : ""}`}
            type="button"
            onClick={() => setActiveTab("env")}
          >
            Environment
          </button>
          <button
            className={`tab-btn ${activeTab === "wandb" ? "active" : ""}`}
            type="button"
            onClick={() => setActiveTab("wandb")}
          >
            W&B Metrics
          </button>
        </div>

        {activeTab === "config" ? (
          <form className="config-grid" onSubmit={handleSubmit}>
          <section className="card">
            <div className="card-head">
              <h2>Test Domain Config</h2>
              <span>Features file content</span>
            </div>
            <div className="inline-controls">
              <div>
                <label htmlFor="preload-config">Preload config</label>
                <select
                  id="preload-config"
                  value={preloadKey}
                  onChange={(event) => handlePreloadChange(event.target.value)}
                >
                  {PRELOADED_CONFIGS.map((config) => (
                    <option key={config.key} value={config.key}>
                      {config.label}
                    </option>
                  ))}
                  <option value="uploaded">Uploaded file (custom)</option>
                </select>
              </div>

              <div>
                <label htmlFor="upload-config">Load local config file</label>
                <input
                  id="upload-config"
                  type="file"
                  accept=".json,.txt"
                  onChange={handleConfigFileUpload}
                />
              </div>
            </div>

            <div className="inline-controls view-controls">
              <div>
                <label htmlFor="domain-view">Domain config view</label>
                <select
                  id="domain-view"
                  value={domainView}
                  onChange={(event) => setDomainView(event.target.value)}
                >
                  <option value="visual">Visual dropdowns and labels</option>
                  <option value="raw">Raw JSON editor</option>
                </select>
              </div>
            </div>

            {domainView === "visual" ? (
              <div className="feature-layout">
                {!parsedFeatureConfig.ok ? (
                  <p className="status error">Cannot parse config JSON: {parsedFeatureConfig.error}</p>
                ) : (
                  <>
                    {["categorical", "ordinal"].map((featureType) => {
                      const sectionFeatures = parsedFeatureConfig.features.filter(
                        (feature) => feature.type === featureType
                      );

                      if (sectionFeatures.length === 0) {
                        return null;
                      }

                      return (
                        <section className="feature-group" key={featureType}>
                          <h3>{featureType === "categorical" ? "Categorical Features" : "Ordinal Features"}</h3>
                          {sectionFeatures.map((feature) => {
                            const selectedIndex = Object.prototype.hasOwnProperty.call(
                              featureSelections,
                              feature.name
                            )
                              ? String(featureSelections[feature.name])
                              : "-1";

                            return (
                              <div className="feature-row" key={`${featureType}-${feature.name}`}>
                                <label htmlFor={`feature-${feature.name}`}>{feature.name}</label>
                                <select
                                  id={`feature-${feature.name}`}
                                  value={selectedIndex}
                                  onChange={(event) =>
                                    handleFeatureSelectionChange(feature.name, event.target.value)
                                  }
                                >
                                  <option value="-1">Not set</option>
                                  {(feature.values || []).map((value, valueIndex) => (
                                    <option key={`${feature.name}-${valueIndex}`} value={String(valueIndex)}>
                                      {formatFeatureValue(value)}
                                    </option>
                                  ))}
                                </select>
                              </div>
                            );
                          })}
                        </section>
                      );
                    })}
                  </>
                )}
              </div>
            ) : (
              <div className="raw-editor">
                <label htmlFor="features-content">Features file content</label>
                <textarea
                  id="features-content"
                  value={form.featuresContent}
                  onChange={(event) => updateField("featuresContent", event.target.value)}
                  rows={14}
                  spellCheck={false}
                />
              </div>
            )}
          </section>

          <section className="card">
            <div className="card-head">
              <h2>Execution Parameters</h2>
              <span>Runtime setup</span>
            </div>

            <label htmlFor="turn-mode">Turn mode</label>
            <select
              id="turn-mode"
              value={form.turnMode}
              onChange={(event) => updateField("turnMode", event.target.value)}
            >
              <option value="single">Single turn</option>
              <option value="multi">Multi turn</option>
            </select>

            <label htmlFor="testing-time">Testing time (minutes)</label>
            <input
              id="testing-time"
              type="number"
              min="1"
              step="1"
              value={form.testingTime}
              onChange={(event) => updateField("testingTime", event.target.value)}
            />

            <label htmlFor="population-size">Population size</label>
            <input
              id="population-size"
              type="number"
              min="1"
              step="1"
              value={form.populationSize}
              onChange={(event) => updateField("populationSize", event.target.value)}
            />
          </section>

          <section className="card">
            <div className="card-head">
              <h2>Evaluation Parameters</h2>
              <span>Fitness and oracle settings</span>
            </div>

            <label htmlFor="judge-model">Judge model</label>
            <input
              id="judge-model"
              type="text"
              value={form.judgeModel}
              onChange={(event) => updateField("judgeModel", event.target.value)}
              placeholder="gpt-4.1-mini"
            />
            <p className="field-note">Predefined in navigation runner: judge is used for response fitness.</p>

            <label htmlFor="judge-temperature">Judge temperature (0.0 - 2.0)</label>
            <input
              id="judge-temperature"
              type="number"
              min="0"
              max="2"
              step="0.05"
              value={form.judgeTemperature}
              onChange={(event) => updateField("judgeTemperature", event.target.value)}
            />
            <p className="field-note">Predefined default in this UI: 0.2</p>

            <div className="fitness-group">
              <div className="fitness-group-head">
                <h3>Fitness Response</h3>
                <label className="toggle-row" htmlFor="fitness-response-enabled">
                  <input
                    id="fitness-response-enabled"
                    type="checkbox"
                    checked={form.fitnessResponseEnabled}
                    onChange={(event) => updateField("fitnessResponseEnabled", event.target.checked)}
                  />
                  <span>Enabled</span>
                </label>
              </div>
              <label htmlFor="fitness-response-name">Predefined value</label>
              <input id="fitness-response-name" type="text" value="answer_fitness" readOnly />
              <p className="field-note">Predefined threshold in navigation runner: 0.75</p>

              <label htmlFor="fitness-response-threshold">Threshold (0.0 - 1.0)</label>
              <input
                id="fitness-response-threshold"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={form.fitnessResponseThreshold}
                onChange={(event) => updateField("fitnessResponseThreshold", event.target.value)}
              />
            </div>

            <div className="fitness-group">
              <div className="fitness-group-head">
                <h3>Fitness Content</h3>
                <label className="toggle-row" htmlFor="fitness-content-enabled">
                  <input
                    id="fitness-content-enabled"
                    type="checkbox"
                    checked={form.fitnessContentEnabled}
                    onChange={(event) => updateField("fitnessContentEnabled", event.target.checked)}
                  />
                  <span>Enabled</span>
                </label>
              </div>
              <label htmlFor="fitness-content-name">Predefined value</label>
              <input id="fitness-content-name" type="text" value="content_fitness" readOnly />
              <p className="field-note">Predefined threshold in navigation runner: 0.75</p>

              <label htmlFor="fitness-content-threshold">Threshold (0.0 - 1.0)</label>
              <input
                id="fitness-content-threshold"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={form.fitnessContentThreshold}
                onChange={(event) => updateField("fitnessContentThreshold", event.target.value)}
              />
            </div>

            <div className="fitness-group">
              <div className="fitness-group-head">
                <h3>Fitness Diverse</h3>
                <label className="toggle-row" htmlFor="fitness-diverse-enabled">
                  <input
                    id="fitness-diverse-enabled"
                    type="checkbox"
                    checked={form.fitnessDiverseEnabled}
                    onChange={(event) => updateField("fitnessDiverseEnabled", event.target.checked)}
                  />
                  <span>Enabled</span>
                </label>
              </div>
              <label htmlFor="fitness-diverse-name">Predefined value</label>
              <input id="fitness-diverse-name" type="text" value="distance" readOnly />
              <p className="field-note">This is the separate `FitnessDiverse()` metric from the navigation runner. It is maximized and the current runner does not define a dedicated threshold for it.</p>
            </div>

            <div className="fitness-group">
              <div className="fitness-group-head">
                <h3>Fitness Efficiency</h3>
                <label className="toggle-row" htmlFor="fitness-efficiency-enabled">
                  <input
                    id="fitness-efficiency-enabled"
                    type="checkbox"
                    checked={form.fitnessEfficiencyEnabled}
                    onChange={(event) => updateField("fitnessEfficiencyEnabled", event.target.checked)}
                  />
                  <span>Enabled</span>
                </label>
              </div>
              <label htmlFor="fitness-efficiency-name">Predefined value</label>
              <input id="fitness-efficiency-name" type="text" value="efficiency_fitness" readOnly />
              <p className="field-note">Additional configurable section in the UI for a backend efficiency metric.</p>

              <label htmlFor="fitness-efficiency-threshold">Threshold (0.0 - 1.0)</label>
              <input
                id="fitness-efficiency-threshold"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={form.fitnessEfficiencyThreshold}
                onChange={(event) => updateField("fitnessEfficiencyThreshold", event.target.value)}
              />
            </div>
          </section>

          <section className="card action-card">
            <div className="card-head">
              <h2>Launch</h2>
              <span>Start backend execution</span>
            </div>

            <label htmlFor="sut">SUT</label>
            <select id="sut" value={form.sut} onChange={(event) => updateField("sut", event.target.value)}>
              <option value="IPA_LOS">IPA_LOS</option>
              <option value="IPA_YELP">IPA_YELP</option>
            </select>

            <label htmlFor="endpoint">Backend endpoint</label>
            <input
              id="endpoint"
              type="url"
              value={form.endpoint}
              onChange={(event) => updateField("endpoint", event.target.value)}
              placeholder="http://localhost:8000/experiments/start"
            />
            <p className="field-note">Use STELLAR runner bridge endpoint, for example: http://localhost:8000/experiments/start</p>

            <label className="toggle-row" htmlFor="enable-wandb">
              <input
                id="enable-wandb"
                type="checkbox"
                checked={form.enableWandb}
                onChange={(event) => updateField("enableWandb", event.target.checked)}
              />
              <span>Enable W&B / Weave logging</span>
            </label>
            <p className="field-note">Recommended off unless you already ran wandb login in the execution environment.</p>

            <button className="start-btn" type="submit" disabled={!canStart || status === "running"}>
              {status === "running" ? "Starting..." : "Start Experiment"}
            </button>

            <p className={`status ${status}`}>{message}</p>
          </section>
          </form>
        ) : activeTab === "runs" ? (
          <section className="card runs-card">
            <div className="card-head">
              <h2>Started Runs</h2>
              <span>Latest first</span>
            </div>

            {runHistory.length === 0 ? (
              <p className="field-note">No runs started yet.</p>
            ) : (
              <div className="runs-list">
                {runHistory.map((run) => (
                  <article className="run-item" key={run.id}>
                    {(() => {
                      const canStop =
                        Boolean(run.backendRunId) && ["queued", "running", "started"].includes(run.status);

                      return (
                    <div className="run-item-head">
                      <strong>{run.sut}</strong>
                      <div className="run-actions">
                        <span className={`run-status ${run.status}`}>{run.status}</span>
                        <button
                          className="secondary-btn stop-btn"
                          type="button"
                          onClick={() => handleStopRun(run)}
                          disabled={!canStop || Boolean(stoppingRuns[run.id])}
                          title={
                            canStop
                              ? "Stop this run"
                              : "Stop is available only for active backend runs"
                          }
                        >
                          {stoppingRuns[run.id] ? "Stopping..." : "Stop"}
                        </button>
                      </div>
                    </div>
                      );
                    })()}
                    <p>
                      Started: {new Date(run.startedAt).toLocaleString()} | Mode: {run.launchMode}
                    </p>
                    <p>
                      Turn: {run.turnMode} | Population: {run.populationSize} | Time: {run.testingTimeMinutes} min
                    </p>
                    <p>Endpoint: {run.endpoint || "not set"}</p>
                    {run.backendRunId ? <p>Backend Run ID: {run.backendRunId}</p> : null}
                    {run.note ? <p>Note: {run.note}</p> : null}
                    <div className="run-console">
                      <p className="run-console-title">Console Output</p>
                      <pre>
                        {(run.logs || [])
                          .map((entry) => `[${new Date(entry.at).toLocaleTimeString()}] ${entry.message}`)
                          .join("\n") || "No output yet."}
                      </pre>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        ) : activeTab === "env" ? (
          <section className="card runs-card">
            <div className="card-head">
              <h2>Environment Variables</h2>
              <span>Run-time overrides</span>
            </div>

            <div className="fitness-group">
              <div className="fitness-group-head">
                <h3>Environment</h3>
                <button className="secondary-btn" type="button" onClick={loadEnvDefaults}>
                  Load from .env
                </button>
              </div>
              <p className="field-note">Defaults are loaded from backend .env. Edit/add KEY=VALUE lines to override for the next run.</p>
              <label htmlFor="env-overrides">Variables (KEY=VALUE per line)</label>
              <textarea
                id="env-overrides"
                value={envText}
                onChange={(event) => setEnvText(event.target.value)}
                rows={16}
                spellCheck={false}
              />
              <p className="field-note">{envStatus || "No .env values loaded yet."}</p>
            </div>
          </section>
        ) : (
          <section className="card runs-card">
            <div className="card-head">
              <h2>W&B Metrics</h2>
              <span>Server-side API fetch</span>
            </div>

            <div className="wandb-controls">
              <div>
                <label htmlFor="wandb-entity">Entity</label>
                <input
                  id="wandb-entity"
                  type="text"
                  value={form.wandbEntity}
                  onChange={(event) => updateField("wandbEntity", event.target.value)}
                  placeholder="opentest"
                />
              </div>
              <div>
                <label htmlFor="wandb-project">Project</label>
                <input
                  id="wandb-project"
                  type="text"
                  value={form.wandbProject}
                  onChange={(event) => updateField("wandbProject", event.target.value)}
                  placeholder="dev"
                />
              </div>
              <div>
                <label htmlFor="wandb-limit">Limit</label>
                <input
                  id="wandb-limit"
                  type="number"
                  min="1"
                  max="100"
                  value={form.wandbLimit}
                  onChange={(event) => updateField("wandbLimit", event.target.value)}
                />
              </div>
              <div className="wandb-button-wrap">
                <button className="secondary-btn" type="button" onClick={loadWandbRuns}>
                  Load Metrics
                </button>
              </div>
            </div>

            <p className="field-note">{wandbStatus}</p>

            {wandbRuns.length === 0 ? (
              <p className="field-note">No W&B runs loaded yet.</p>
            ) : (
              <div className="wandb-list">
                {wandbRuns.map((run) => (
                  <article className="wandb-item" key={run.id}>
                    <div className="run-item-head">
                      <strong>{run.name || run.id}</strong>
                      <span className={`run-status ${run.state || "running"}`}>{run.state || "unknown"}</span>
                    </div>
                    <p>Created: {run.createdAt ? new Date(run.createdAt).toLocaleString() : "n/a"}</p>
                    <p>Run URL: {run.url || "n/a"}</p>
                    <pre>{JSON.stringify(run.summary || {}, null, 2)}</pre>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}

      </main>
    </div>
  );
}

export default App;
