/* The form edits answers; Python remains the authoritative builder and validator. */
"use strict";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const icon = (name) => `<i data-lucide="${name}"></i>`;
const state = {
  bootstrap: null,
  answers: null,
  step: 0,
  saved: null,
  preview: null,
  file: "config.yaml",
  view: "configuration",
  selectedJob: null,
  logSource: "console",
  jobs: [],
  busy: false,
  draft: null,
  dirty: false,
  issues: [],
  example: null,
};
const fragment = new URLSearchParams(location.hash.slice(1));
if (fragment.has("token")) {
  sessionStorage.setItem("harborrl-token", fragment.get("token"));
  history.replaceState(null, "", location.pathname);
}
const token = sessionStorage.getItem("harborrl-token") || "";
function icons() {
  lucide.createIcons({ attrs: { "aria-hidden": "true" } });
}
function notify(text, success = false) {
  $("#notice").textContent = text;
  $("#notice").className = `notice${success ? " success" : ""}`;
  $("#notice").hidden = false;
}
async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const result = await response.json();
  if (!response.ok) {
    const error = new Error(result.error || "Request failed");
    error.fields = result.fields || [];
    throw error;
  }
  return result;
}
async function action(callback) {
  if (state.busy) return;
  state.busy = true;
  updateActions();
  $("#notice").hidden = true;
  try {
    await callback();
  } catch (error) {
    notify(error.message);
  } finally {
    state.busy = false;
    updateActions();
    icons();
  }
}
function valueAt(object, path) {
  return path.split(".").reduce((value, key) => value?.[key], object);
}
function setAt(object, path, value) {
  const parts = path.split(".");
  let parent = object;
  parts.slice(0, -1).forEach((key) => {
    if (!parent[key] || typeof parent[key] !== "object") parent[key] = {};
    parent = parent[key];
  });
  parent[parts.at(-1)] = value;
}
function defaultRoot() {
  const separator = state.bootstrap.workspace.includes("\\") ? "\\" : "/";
  return [
    state.bootstrap.workspace,
    "configurations",
    $("#config-name").value,
    "runs",
  ].join(separator);
}
function read(path) {
  const supplied = valueAt(state.answers, path);
  if (supplied !== undefined) return supplied;
  if (path === "local_environment.source_dir") return read("deployment.manager.source_dir");
  if (path === "local_environment.python") return read("deployment.manager.controller_python");
  if (path === "sandbox_config")
    return valueAt(state.answers, "training.harbor.environment.config");
  if (path === "deployment.storage.run_root")
    return valueAt(state.answers, "training.output.root") ?? defaultRoot();
  if (
    [
      "training.deployment.actor_gpus",
      "training.deployment.rollout_gpus",
    ].includes(path) &&
    read("training.deployment.layout") === "colocate"
  )
    return read("training.deployment.num_gpus");
  return valueAt(state.bootstrap.defaults[state.answers.preset], path);
}
function changed() {
  state.saved = null;
  state.preview = null;
  state.dirty = true;
  $("#draft-state").textContent = "Unsaved changes";
  $("#draft-state").className = "badge";
  updateActions();
  renderSummary();
}
function deriveWorkers() {
  if (state.answers.preset === "service-ssh") {
    delete state.answers.training.harbor?.workers;
    delete state.answers.training.harbor?.python;
  }
}
function newAnswers(preset = "embedded-local") {
  const answers = { answers_version: 1, preset, training: {}, deployment: {} };
  if (preset === "embedded-local" && state.bootstrap) {
    const manager = state.bootstrap.defaults[preset].deployment.manager;
    answers.local_environment = { source_dir: manager.source_dir, python: manager.controller_python };
  }
  return answers;
}
function field(path, label, type = "text", options = {}) {
  const value = read(path),
    id = path.replaceAll(".", "-");
  const help = state.bootstrap.field_help[path]?.help || options.hint ||
    (path.includes("workers.") ? workerHelp(path) : "");
  const invalid = state.issues.some((issue) => issue.path === path);
  const attributes = `aria-describedby="${id}-help" aria-invalid="${invalid}" ${options.required ? 'aria-required="true" required' : ""}`;
  let control;
  if (options.choices)
    control = `<select id="${id}" data-path="${path}" ${attributes}>${options.choices.map((item) => `<option value="${item}" ${value === item ? "selected" : ""}>${item}</option>`).join("")}</select>`;
  else if (type === "checkbox")
    control = `<input id="${id}" data-path="${path}" type="checkbox" ${value ? "checked" : ""}>`;
  else
    control = `<input id="${id}" data-path="${path}" type="${type}" value="${escapeHtml(value)}" ${type === "number" ? 'min="1" step="1"' : ""} ${options.step ? `step="${options.step}"` : ""} placeholder="${escapeHtml(options.placeholder || "")}" autocomplete="off" ${attributes}>`;
  return `<div class="field ${invalid ? "invalid-field" : ""} ${options.full ? "full" : ""} ${options.required ? "required" : ""} ${type === "checkbox" ? "checkbox-field" : ""}">${type === "checkbox" ? control : ""}<label for="${id}">${label}</label>${type === "checkbox" ? "" : control}<small id="${id}-help">${escapeHtml(help)}</small></div>`;
}
function workerHelp(path) {
  if (path.endsWith("ssh.host")) return "SSH alias reachable from the controller.";
  if (path.endsWith("source_dir")) return "Absolute HarborRL checkout directory on this worker.";
  if (path.endsWith("runner_python")) return "Absolute Python executable with Harbor installed on this worker.";
  if (path.includes("task_roots")) return "Absolute directory containing the task files on this worker.";
  return "Unique worker name using letters, numbers and hyphens.";
}
const section = (title, content) =>
  `<section class="form-section"><div class="section-heading"><h2>${title}</h2></div><div class="form-grid">${content}</div></section>`;
function renderForm() {
  const preset = state.answers.preset,
    service = preset !== "embedded-local",
    sandbox = preset === "service-sandbox-local-runner";
  let html = "";
  if (state.step === 0) {
    const presets = [
      ["embedded-local", "Local", "One machine", "monitor"],
      ["service-ssh", "SSH workers", "Distributed Docker", "network"],
      ["service-sandbox-local-runner", "Sandbox", "Controller + GPU", "box"],
    ];
    html = `<div class="section-heading"><h2>Deployment topology</h2><span class="badge">${service ? "Service" : "Embedded"}</span></div><div class="preset-options">${presets.map(([id, label, detail, ico]) => `<button class="preset ${preset === id ? "active" : ""}" data-preset="${id}" aria-pressed="${preset === id}">${icon(ico)}<div>${label}<small>${detail}</small></div></button>`).join("")}</div>`;
    if (!service) html += section(
      "Shared local environment",
      '<p class="full muted">Controller, GPU and rollout share this source directory and Python. Values were detected on the machine running HarborRL. Choose an environment with the required training and Harbor dependencies.</p>' +
      field("local_environment.source_dir", "Shared source directory", "text", {full: true, required: true}) +
      field("local_environment.python", "Shared Python interpreter", "text", {full: true, required: true}),
    );
    if (service) html += section(
      "Controller",
      field("deployment.manager.source_dir", "Source directory", "text", {
        full: true,
        hint: "Path on the controller",
        required: true,
      }) +
        field(
          "deployment.manager.controller_python",
          "Controller Python",
          "text",
          { full: true, required: true },
        ) +
        (service
          ? field("deployment.manager.control_url", "Control origin", "url", {
              placeholder: "http://controller.example:49200",
              required: true,
            }) +
            field("deployment.manager.gateway_url", "Gateway origin", "url", {
              placeholder: "http://controller.example:49184",
              required: true,
            })
          : ""),
    );
    if (service) html += section(
      "GPU target",
      (service
        ? field("deployment.gpu_target.ssh.host", "SSH destination", "text", {
            placeholder: "gpu-training",
            full: true,
            required: true,
          })
        : "") +
        field("deployment.gpu_target.source_dir", "Source directory", "text", {
          full: true,
          required: true,
          hint: "Path on the GPU machine",
        }) +
        field("deployment.gpu_target.train_python", "Training Python", "text", {
          full: true,
          required: true,
        }),
    );
    if (preset === "service-ssh")
      html += `<section class="form-section"><div class="section-heading"><h2>Rollout workers</h2><button id="add-worker" class="secondary">${icon("plus")}Add worker</button></div><div id="worker-fields"></div></section>`;
    else if (service)
      html += section(
        "Rollout runner",
        field("training.harbor.python", "Harbor runner Python", "text", {
          full: true,
          required: true,
        }) +
          (sandbox
            ? field("sandbox_config", "Sandbox configuration", "text", {
                full: true,
                placeholder: "/srv/site/sandbox.yaml",
                required: true,
              })
            : ""),
      );
    html += section(
      "Storage",
      field(
        "deployment.storage.run_root",
        service ? "Shared run root" : "Run root",
        "text",
        {
          full: true,
          required: true,
          hint: service
            ? "Same absolute path on the controller, GPU and workers"
            : "Path on this machine",
        },
      ),
    );
  } else if (state.step === 1) {
    html += section(
      "Model",
      field("training.model.args_file", "Model args preset", "text", {
        required: true,
      }) +
        field("training.harness.name", "Harness", "text", {
          choices: state.bootstrap.catalog.harnesses.map((x) => x.id),
        }) +
        field("training.model.checkpoint", "Checkpoint directory", "text", {
          full: true,
          required: true,
          hint: "Path on the GPU target",
        }) +
        field("training.model.reference", "Reference checkpoint", "text", {
          full: true,
          required: true,
        }),
    );
    html += section(
      "Tasks & agent",
      '<div class="full input-starter"><button id="hello-inputs" class="secondary">Use bundled Hello World task & profile</button><p class="muted">Creates a task catalog and agent profile on this machine. Training still needs your model and runtime.</p></div>' +
      field("training.tasks.catalog", "Task catalog", "text", {
        full: true,
        required: true,
        placeholder: "/srv/inputs/tasks.jsonl",
        hint: "An existing locked JSON / JSONL catalog on the controller",
      }) +
        field("training.harness.profile", "Claude Code profile", "text", {
          full: true,
          required: true,
          placeholder: "/srv/inputs/claude-code.json",
        }),
    );
    html += section(
      "Gateway fingerprints",
      `<div class="full model-preparation" id="model-preparation"><p>Calculated automatically from the actual checkpoint tokenizer using the GPU target Python${service ? " over SSH" : ""}. No SHA-256 entry is needed.</p><button id="prepare-model" class="secondary">${icon("scan-line")}Prepare model</button><strong id="fingerprint-status">${read("training.gateway.tokenizer_digest") && read("training.gateway.template_digest") ? "Fingerprints prepared" : "Model preparation required"}</strong><details><summary>Calculated fingerprints</summary><dl><dt>Tokenizer</dt><dd>${escapeHtml(read("training.gateway.tokenizer_digest") || "Not calculated")}</dd><dt>Chat template</dt><dd>${escapeHtml(read("training.gateway.template_digest") || "Not calculated")}</dd></dl></details></div>` +
        (!service
          ? field("training.gateway.host", "Bind address") +
            field("training.gateway.advertised_url", "Advertised origin", "url", {required: true})
          : ""),
    );
  } else if (state.step === 2) {
    html += section(
      "GPU allocation",
      field("training.deployment.layout", "Layout", "text", {
        choices: ["colocate", "split"],
      }) +
        field("training.deployment.num_gpus", "Total GPUs", "number") +
        field("training.deployment.actor_gpus", "Actor GPUs", "number") +
        field("training.deployment.rollout_gpus", "Serving GPUs", "number") +
        field(
          "training.deployment.actor_tensor_parallel_size",
          "Actor tensor parallelism",
          "number",
        ) +
        field(
          "training.deployment.rollout_gpus_per_engine",
          "Serving tensor parallelism",
          "number",
        ),
    );
    html += section(
      "Optimization",
      field("training.training.num_rollout", "Rollout batches", "number") +
        field("training.training.learning_rate", "Learning rate", "text") +
        field("training.training.save_interval", "Save interval", "number") +
        field(
          "training.training.backend_contract",
          "Algorithm contract",
          "text",
          {
            choices: [
              ...new Set(
                state.bootstrap.catalog.algorithms.flatMap(
                  (x) => x.training_backend_contracts,
                ),
              ),
            ],
          },
        ),
    );
    html += section(
      "Sampling",
      field("training.sampling.group_size", "Samples per group", "number") +
        field(
          "training.sampling.groups_per_batch",
          "Groups per batch",
          "number",
        ) +
        field(
          "training.sampling.max_tokens",
          "Maximum output tokens",
          "number",
        ) +
        field("training.sampling.max_context", "Context limit", "number") +
        field("training.sampling.max_attempts", "Maximum attempts", "number") +
        field(
          "training.gateway.audited_raw_logprobs",
          "Audited raw log probabilities",
          "checkbox",
        ),
    );
  }
  $("#form-sections").innerHTML = html;
  $("#form-sections").hidden = state.step === 3;
  $("#review").hidden = state.step !== 3;
  if (state.step === 0 && preset === "service-ssh") renderWorkers();
  $$("[data-path]").forEach((input) =>
    input.addEventListener("change", () => {
      const path = input.dataset.path;
      let value =
        input.type === "checkbox"
          ? input.checked
          : input.type === "number" || path.endsWith("learning_rate")
            ? input.value === ""
              ? null
              : Number(input.value)
            : input.value;
      if (path.startsWith("local_environment.")) {
        state.answers.local_environment = {
          source_dir: read("local_environment.source_dir"),
          python: read("local_environment.python"),
        };
      }
      setAt(state.answers, path, value);
      state.issues = state.issues.filter((issue) => issue.path !== path);
      input.setAttribute("aria-invalid", "false");
      input.closest(".field").classList.remove("invalid-field");
      renderIssues();
      if (["training.model.checkpoint", "local_environment.python", "local_environment.source_dir",
        "deployment.gpu_target.train_python", "deployment.gpu_target.source_dir", "deployment.gpu_target.ssh.host"].includes(path)) {
        if (state.answers.training.gateway) {
          delete state.answers.training.gateway.tokenizer_digest;
          delete state.answers.training.gateway.template_digest;
        }
        if ($("#fingerprint-status")) $("#fingerprint-status").textContent = "Checkpoint or environment changed; prepare model again";
      }
      // Loaded bundles contain both sides of repeated schema fields.
      const mirrors = {
        "deployment.manager.control_url": null,
        "deployment.manager.gateway_url": "training.gateway.advertised_url",
        "deployment.storage.run_root": "training.output.root",
        "training.gateway.host": "deployment.manager.bind",
        "training.gateway.advertised_url": "deployment.manager.gateway_url",
      };
      if (mirrors[path] && valueAt(state.answers, mirrors[path]) !== undefined)
        setAt(state.answers, mirrors[path], value);
      if (
        path === "sandbox_config" &&
        valueAt(state.answers, "training.harbor.environment.config") !==
          undefined
      )
        setAt(state.answers, "training.harbor.environment.config", value);
      if (
        path.startsWith("deployment.rollout.workers.") &&
        state.answers.preset === "service-ssh"
      ) {
        deriveWorkers();
      }
      if (path === "training.deployment.num_gpus") {
        if (
          valueAt(state.answers, "deployment.gpu_target.expected_gpus") !==
          undefined
        )
          setAt(state.answers, "deployment.gpu_target.expected_gpus", value);
        if (read("training.deployment.layout") === "colocate")
          ["actor_gpus", "rollout_gpus"].forEach((key) =>
            setAt(state.answers, `training.deployment.${key}`, value),
          );
      }
      if (path === "training.deployment.layout" && value === "colocate") {
        ["actor_gpus", "rollout_gpus"].forEach((key) =>
          setAt(
            state.answers,
            `training.deployment.${key}`,
            read("training.deployment.num_gpus"),
          ),
        );
      }
      changed();
      if (path.startsWith("training.deployment.")) renderForm();
    }),
  );
  $$("[data-preset]").forEach((button) =>
    button.addEventListener("click", () =>
      action(async () => {
        if (button.dataset.preset === state.answers.preset) return;
        if (
          !(await confirmAction(
            "Change deployment?",
            "This starts a new draft for the selected topology. Saved configurations remain available.",
            "Change topology",
          ))
        )
          return;
        state.answers = newAnswers(button.dataset.preset);
        state.draft = null;
        state.issues = [];
        changed();
        renderForm();
      }),
    ),
  );
  $("#prepare-model")?.addEventListener("click", () => action(async () => {
    $("#fingerprint-status").textContent = "Loading tokenizer and calculating fingerprints...";
    try {
      const result = await api("/api/prepare-model", {name: $("#config-name").value, answers: state.answers});
      setAt(state.answers, "training.gateway.tokenizer_digest", result.fingerprints.tokenizer_digest);
      setAt(state.answers, "training.gateway.template_digest", result.fingerprints.template_digest);
      state.issues = state.issues.filter((issue) => !issue.path.includes("_digest"));
      changed();
      renderForm();
      renderIssues();
      notify(`Model fingerprints calculated on ${result.target}.`, true);
    } catch (error) {
      $("#fingerprint-status").textContent = "Preparation failed. Check the environment and checkpoint; you can save this draft.";
      throw error;
    }
  }));
  $("#hello-inputs")?.addEventListener("click", () => action(async () => {
    const inputs = await api("/api/hello-inputs", {});
    setAt(state.answers, "training.tasks.catalog", inputs.catalog);
    setAt(state.answers, "training.harness.profile", inputs.profile);
    state.issues = state.issues.filter((issue) => !["training.tasks.catalog", "training.harness.profile"].includes(issue.path));
    changed(); renderForm();
    notify("Bundled task catalog and agent profile are ready.", true);
  }));
  $("#add-worker")?.addEventListener("click", () => {
    const workers = state.answers.deployment.rollout?.workers || [];
    workers.push({
      id: `worker-${Date.now().toString(36)}`,
      ssh: { host: "" },
      source_dir: "",
      runner_python: "",
      docker: true,
      task_roots: [""],
    });
    setAt(state.answers, "deployment.rollout.workers", workers);
    deriveWorkers();
    changed();
    renderForm();
  });
  renderSummary();
  renderIssues();
  updateActions();
  icons();
}
function renderWorkers() {
  const workers = state.answers.deployment.rollout?.workers || [];
  $("#worker-fields").innerHTML = workers.length
    ? workers
        .map(
          (worker, index) =>
            `<div class="worker-group"><div class="worker-heading"><h3>Worker ${index + 1}</h3><button class="icon-button" data-remove-worker="${index}" title="Remove worker" aria-label="Remove worker ${index + 1}">${icon("trash-2")}</button></div><div class="form-grid">${field(`deployment.rollout.workers.${index}.id`, "ID", "text", {required: true})}${field(`deployment.rollout.workers.${index}.ssh.host`, "SSH destination", "text", {required: true})}${field(`deployment.rollout.workers.${index}.source_dir`, "Source directory", "text", { full: true , required: true })}${field(`deployment.rollout.workers.${index}.runner_python`, "Runner Python", "text", { full: true , required: true })}${field(`deployment.rollout.workers.${index}.task_roots.0`, "Task root", "text", { full: true , required: true })}</div></div>`,
        )
        .join("")
    : '<p class="muted">No workers configured.</p>';
  $$("[data-remove-worker]").forEach((button) =>
    button.addEventListener("click", () => {
      workers.splice(Number(button.dataset.removeWorker), 1);
      deriveWorkers();
      changed();
      renderForm();
    }),
  );
}
function renderSummary() {
  const service = state.answers.preset !== "embedded-local",
    sandbox = state.answers.preset === "service-sandbox-local-runner";
  const nodes = [
    ["monitor", "Controller", service ? "Service mode" : "Embedded mode"],
    [
      "cpu",
      "GPU target",
      service
        ? read("deployment.gpu_target.ssh.host") || "Not configured"
        : "Local machine",
    ],
    [
      sandbox ? "box" : "network",
      "Rollout",
      sandbox
        ? "Sandbox environment"
        : service
          ? `${state.answers.deployment.rollout?.workers?.length || 0} SSH workers`
          : "Local Docker",
    ],
  ];
  $("#topology").innerHTML = nodes
    .map(
      ([ico, title, detail], index) =>
        `${index ? '<div class="topology-connector"></div>' : ""}<div class="topology-node"><div class="node-icon">${icon(ico)}</div><div><strong>${title}</strong><span>${escapeHtml(detail)}</span></div></div>`,
    )
    .join("");
  const rows = [
    ["Model", read("training.model.args_file")],
    ["Algorithm", "GRPO"],
    ["GPUs", read("training.deployment.num_gpus")],
    ["Layout", read("training.deployment.layout")],
    [
      "Samples / batch",
      Number(read("training.sampling.group_size")) *
        Number(read("training.sampling.groups_per_batch")),
    ],
    ["Rollout batches", read("training.training.num_rollout")],
  ];
  $("#resource-summary").innerHTML =
    `<dl>${rows.map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(value)}</dd>`).join("")}</dl>`;
  $("#summary-status").textContent = state.saved
    ? "Saved configuration"
    : state.draft && !state.dirty ? "Draft saved; continue when ready" : "Unsaved changes";
  icons();
}
function updateActions() {
  $("#previous").disabled = state.step === 0 || state.busy;
  $("#next").hidden = state.step === 3;
  $("#save").hidden = state.step !== 3;
  $("#next").disabled = state.busy;
  $("#save").disabled =
    state.busy ||
    !state.preview ||
    state.preview.validation.status === "invalid" ||
    !!state.saved;
  $("#save-draft").disabled = state.busy;
  $("#save-draft").textContent = state.draft && !state.dirty ? "Draft saved" : "Save draft";
  $("#prepare-model")?.toggleAttribute("disabled", state.busy);
  $("#step-description").textContent = `Step ${state.step + 1} of 4`;
  $$("[data-step]").forEach((button) => {
    button.classList.toggle(
      "active",
      Number(button.dataset.step) === state.step,
    );
    button.setAttribute(
      "aria-current",
      Number(button.dataset.step) === state.step ? "step" : "false",
    );
  });
  const active = state.jobs.some((job) =>
    ["running", "stopping", "unknown"].includes(job.state),
  );
  $("#hello-world").disabled = state.busy || active;
  $("#dry-run").disabled = !state.saved || state.busy || active;
  $("#doctor").disabled =
    !state.saved || state.busy || active || !state.bootstrap?.live_supported;
  $("#train").disabled =
    !state.saved || state.busy || active || !state.bootstrap?.live_supported;
}
async function preview(checkFiles = false) {
  state.preview = null;
  $("#validation-result").textContent = "Checking configuration...";
  $("#config-preview").textContent = "";
  try {
    state.preview = await api("/api/preview", {
      name: $("#config-name").value,
      answers: state.answers,
      check_files: checkFiles,
    });
    const result = state.preview.validation;
    state.issues = [];
    renderIssues();
    $("#validation-result").innerHTML =
      `<div class="check-status">${result.status === "invalid" ? "Changes required" : "Structure valid · runtime checks deferred"}</div>${result.checks.map((row) => `<div class="validation-row ${row.status}">${icon(row.status === "passed" ? "circle-check" : row.status === "failed" ? "circle-x" : "clock-3")}<div><strong>${escapeHtml(row.path)}</strong><p>${escapeHtml(row.message)}</p></div></div>`).join("")}`;
    renderPreview();
  } catch (error) {
    state.issues = error.fields || [];
    renderIssues();
    $("#validation-result").textContent = state.issues.length
      ? `${state.issues.length} required fields need attention. Follow the links above, or save this draft and return later.`
      : error.message;
    if (!state.issues.length) throw error;
  } finally {
    updateActions();
    icons();
  }
}
function renderIssues() {
  const panel = $("#missing-fields");
  panel.hidden = !state.issues.length;
  panel.innerHTML = `<strong>${state.issues.length} required fields need attention</strong><p>You can save an incomplete draft at any time.</p><ul>${state.issues.map((issue, index) => `<li><button data-issue="${index}"><strong>${escapeHtml(issue.label)}</strong><span>${escapeHtml(issue.help)}</span></button></li>`).join("")}</ul>`;
  $$("[data-issue]").forEach((button) => button.addEventListener("click", () => action(async () => {
    const issue = state.issues[Number(button.dataset.issue)];
    state.step = issue.step; renderForm();
    let path = issue.path;
    if (state.answers.preset === "embedded-local") {
      if (path === "deployment.manager.source_dir") path = "local_environment.source_dir";
      if (path === "deployment.manager.controller_python" || path === "training.harbor.python") path = "local_environment.python";
    }
    const target = path.includes("_digest") ? $("#prepare-model") :
      document.getElementById(path.replaceAll(".", "-")) || $("#add-worker");
    if (target) {target.focus(); target.scrollIntoView({block: "center", behavior: "smooth"});}
  })));
}
function renderPreview() {
  $("#config-preview").textContent = state.preview?.files[state.file] || "";
  $$("[data-file]").forEach((button) =>
    button.classList.toggle("active", button.dataset.file === state.file),
  );
}
async function goStep(step) {
  state.step = step;
  renderForm();
  if (step === 3) await preview();
}
function showView(view) {
  state.view = view;
  $("#configuration-view").hidden = view !== "configuration";
  $("#runs-view").hidden = view !== "runs";
  $$("[data-view]").forEach((button) =>
    button.classList.toggle("active", button.dataset.view === view),
  );
  if (view === "runs") renderJobs();
}
function renderSaved() {
  const configurations = [...(state.bootstrap.drafts || []).map((item) => ({...item, kind: "draft"})),
    ...state.bootstrap.configurations];
  $("#saved-list").innerHTML = configurations.length
    ? configurations
        .map(
          (config) =>
            `<button class="saved-item ${state.saved === config.id ? "selected" : ""}" data-config="${escapeHtml(config.id)}" data-kind="${config.kind || "configuration"}">${icon(config.kind === "draft" ? "file-pen-line" : "file-check")}<span>${escapeHtml(config.id)}<small>${config.kind === "draft" ? "Draft" : "Runnable"}</small></span></button>`,
        )
        .join("")
    : '<span class="muted">No configurations yet</span>';
  $$("[data-config]").forEach((button) =>
    button.addEventListener("click", () =>
      action(async () => {
        if (
          state.dirty &&
          !(await confirmAction(
            "Open saved configuration?",
            "The current draft will be replaced.",
            "Open configuration",
          ))
        )
          return;
        const draft = button.dataset.kind === "draft";
        const config = await api(
          `/api/${draft ? "drafts" : "configurations"}/${encodeURIComponent(button.dataset.config)}`,
        );
        state.answers = config.answers;
        state.draft = draft ? {id: config.id, revision: config.revision} : null;
        state.dirty = false; state.issues = [];
        state.saved = draft ? null : config.id;
        $("#config-name").value = config.id;
        $("#draft-state").textContent = draft ? "Draft saved" : "Runnable";
        $("#draft-state").className = "badge saved";
        showView("configuration");
        await goStep(draft ? 0 : 3);
        renderSaved();
      }),
    ),
  );
  icons();
}
function download(name, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function confirmAction(title, message, label = "Confirm") {
  const dialog = $("#confirm-dialog");
  $("#confirm-title").textContent = title;
  $("#confirm-message").textContent = message;
  $("#confirm-action").textContent = label;
  dialog.returnValue = "cancel";
  return new Promise((resolve) => {
    dialog.addEventListener(
      "close",
      () => resolve(dialog.returnValue === "confirm"),
      { once: true },
    );
    dialog.showModal();
  });
}
async function startJob(mode) {
  if (
    mode !== "dry-run" &&
    !(await confirmAction(
      mode === "train" ? "Start training?" : "Run live environment checks?",
      mode === "train"
        ? `Configuration: ${state.saved}\nGPU target: ${read("deployment.gpu_target.ssh.host") || "Local machine"}\nGPU budget: ${read("training.deployment.num_gpus")}\n\nTraining runs preflight, then allocates the configured GPU and rollout resources.`
        : "Doctor executes environment and network checks on the configured machines.",
      mode === "train" ? "Start training" : "Run doctor",
    ))
  )
    return;
  const job = await api("/api/jobs", {
    configuration: state.saved,
    mode,
    confirmed: mode !== "dry-run",
  });
  state.selectedJob = job.id;
  showView("runs");
  await refreshJobs();
}
function renderJobs() {
  $("#run-count").textContent = state.jobs.length;
  $("#jobs-list").innerHTML = state.jobs.length
    ? `<div class="table-wrap"><table class="jobs-table"><thead><tr><th>CONFIGURATION</th><th>MODE</th><th>STARTED</th><th>STATUS</th><th>EXIT</th></tr></thead><tbody>${state.jobs.map((job) => `<tr data-job="${job.id}" tabindex="0" class="${job.id === state.selectedJob ? "selected" : ""}"><td><div class="job-name">${escapeHtml(job.configuration)}</div><div class="job-id">${job.id.slice(0, 8)}</div></td><td>${escapeHtml(job.mode)}</td><td>${new Date(job.created_at).toLocaleString()}</td><td><span class="badge ${job.state}">${escapeHtml(job.state)}</span></td><td>${job.exit_code ?? "—"}</td></tr>`).join("")}</tbody></table></div>`
    : `<div class="empty-state">${icon("activity")}<h2>No runs yet</h2><p>Your dry-runs, environment checks and training jobs appear here.</p></div>`;
  $$("[data-job]").forEach((row) => {
    const select = () =>
      action(async () => {
        state.selectedJob = row.dataset.job;
        renderJobs();
        await loadLog();
      });
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter") select();
    });
  });
  icons();
}
async function loadLog() {
  if (!state.selectedJob) return;
  const result = await api(`/api/jobs/${state.selectedJob}`),
    job = result.job;
  $("#job-detail").hidden = false;
  $("#log-selector").hidden = false;
  $("#job-title").textContent = `${job.configuration} · ${job.mode}`;
  $("#job-meta").textContent =
    `${job.state} · ${job.snapshot}${job.message ? ` · ${job.message}` : ""}`;
  const log = $("#job-log"),
    atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 25;
  const training = state.logSource === "training";
  const logError = training ? result.training_log_error : result.log_error;
  log.textContent =
    (training ? result.training_text : result.text) ||
    (logError ? `Log unavailable: ${logError}` : "Waiting for process output...");
  if (atBottom) log.scrollTop = log.scrollHeight;
  $("#log-note").textContent =
    `${(training ? result.training_truncated : result.truncated) ? "Showing the last 64 KiB. " : ""}${training ? result.training_log_path || "No separate GPU log yet" : job.log_path} · Output: ${job.output_root}`;
  $("#stop-job").hidden = !["running", "stopping"].includes(job.state);
  $("#stop-job").disabled = job.state === "stopping";
  $("#reconcile-job").hidden = job.state !== "unknown";
}
async function refreshJobs() {
  state.jobs = await api("/api/jobs");
  renderJobs();
  updateActions();
  if (state.view === "runs") await loadLog();
}
async function importAnswers(text) {
  const { answers } = await api("/api/import", { text });
  state.answers = answers;
  state.draft = null; state.issues = [];
  changed();
  state.step = 0;
  renderForm();
  showView("configuration");
  notify("Answers imported", true);
}
async function initialize() {
  icons();
  try {
    state.bootstrap = await api("/api/bootstrap");
    state.jobs = state.bootstrap.jobs;
    state.answers = newAnswers();
    $("#workspace-path").textContent = state.bootstrap.workspace;
    $("#connection-text").textContent = "Local session";
    renderForm();
    renderSaved();
    renderJobs();
    if (!state.bootstrap.live_supported)
      notify(
        "Configuration and dry-run are available here. Start this console on the Linux controller for live training.",
      );
  } catch (error) {
    $("#connection-text").textContent = "Disconnected";
    $(".connection").classList.add("offline");
    notify(error.message);
    return;
  }
  $$("[data-view]").forEach((button) =>
    button.addEventListener("click", () => showView(button.dataset.view)),
  );
  $$("[data-step]").forEach((button) =>
    button.addEventListener("click", () =>
      action(() => goStep(Number(button.dataset.step))),
    ),
  );
  $("#previous").addEventListener("click", () =>
    action(() => goStep(state.step - 1)),
  );
  $("#next").addEventListener("click", () =>
    action(() => goStep(state.step + 1)),
  );
  $("#config-name").addEventListener("change", () => {
    changed();
    if (state.step === 3) action(() => preview());
    else renderForm();
  });
  $("#check-files").addEventListener("click", () =>
    action(() => preview(true)),
  );
  $$("[data-file]").forEach((button) =>
    button.addEventListener("click", () => {
      state.file = button.dataset.file;
      renderPreview();
    }),
  );
  $("#download-file").addEventListener("click", () => {
    if (state.preview) download(state.file, state.preview.files[state.file]);
  });
  $("#save").addEventListener("click", () =>
    action(async () => {
      // Runnable bundles are immutable. Publish edits as a new numbered version.
      const baseName = $("#config-name").value;
      let name = baseName, index = 2;
      while (state.bootstrap.configurations.some((item) => item.id === name)) name = `${baseName.slice(0, 56)}-v${index++}`;
      const saved = await api("/api/configurations", {
        name,
        answers: state.answers,
      });
      state.saved = saved.id;
      state.dirty = false;
      state.bootstrap.configurations.push(saved);
      $("#draft-state").textContent = "Runnable";
      $("#draft-state").className = "badge saved";
      renderSaved();
      renderSummary();
      notify(`Saved ${saved.id}`, true);
    }),
  );
  $("#save-draft").addEventListener("click", () => action(async () => {
    const name = $("#config-name").value;
    const saved = await api("/api/drafts", {name, answers: state.answers,
      revision: state.draft?.id === name ? state.draft.revision : null});
    state.draft = {id: saved.id, revision: saved.revision};
    state.dirty = false;
    state.bootstrap.drafts = [...state.bootstrap.drafts.filter((item) => item.id !== saved.id), saved];
    $("#draft-state").textContent = "Draft saved";
    $("#draft-state").className = "badge saved";
    renderSaved(); renderSummary();
    notify(`Draft saved: ${saved.id}. Reopen it from the sidebar to continue.`, true);
  }));
  $("#hello-world").addEventListener("click", () => action(async () => {
    const job = await api("/api/hello", {});
    state.selectedJob = job.id; state.logSource = "console"; $("#log-source").value = "console";
    showView("runs"); await refreshJobs();
  }));
  $$("[data-example]").forEach((button) => button.addEventListener("click", () => {
    state.example = state.bootstrap.examples.find((item) => item.id === button.dataset.example);
    $("#example-title").textContent = state.example.title;
    $("#example-notes").innerHTML = state.example.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");
    $("#example-content").textContent = JSON.stringify(state.example.answers, null, 2);
    $("#example-dialog").showModal();
  }));
  $("#close-example").addEventListener("click", () => $("#example-dialog").close());
  $("#download-example").addEventListener("click", () => download(`${state.example.id}-answers.json`, JSON.stringify(state.example.answers, null, 2), "application/json"));
  $("#apply-example").addEventListener("click", () => action(async () => {
    const example = structuredClone(state.example);
    $("#example-dialog").close();
    if (state.dirty && !(await confirmAction("Use this example?", "Replace the current unsaved answers with the reference recipe?", "Use example"))) return;
    state.answers = example.answers; state.draft = null; state.issues = [];
    $("#config-name").value = `${example.id === "embedded-local" ? "local" : "sandbox"}-${Date.now().toString(36)}`;
    state.step = 0; changed(); renderForm();
    notify("Example loaded as a draft. Replace example paths and hosts, use the bundled task inputs, then prepare your model.", true);
  }));
  ["dry-run", "doctor", "train"].forEach((mode) =>
    $("#" + mode).addEventListener("click", () => action(() => startJob(mode))),
  );
  $("#new-config").addEventListener("click", () =>
    action(async () => {
      if (
        !(await confirmAction(
          "New configuration?",
          "Start a fresh draft?",
          "New configuration",
        ))
      )
        return;
      state.answers = newAnswers();
      state.draft = null; state.issues = [];
      $("#config-name").value =
        `experiment-${state.bootstrap.configurations.length + 1}`;
      state.step = 0;
      changed();
      showView("configuration");
      renderForm();
      renderSaved();
    }),
  );
  $("#answers-button").addEventListener("click", () => {
    $("#answers-error").hidden = true;
    $("#answers-editor").value = JSON.stringify(state.answers, null, 2);
    $("#answers-dialog").showModal();
  });
  $("#close-answers").addEventListener("click", () =>
    $("#answers-dialog").close(),
  );
  $("#apply-answers").addEventListener("click", () =>
    action(async () => {
      try {
        await importAnswers($("#answers-editor").value);
        $("#answers-dialog").close();
      } catch (error) {
        $("#answers-error").textContent = error.message;
        $("#answers-error").hidden = false;
      }
    }),
  );
  $("#mobile-new").addEventListener("click", () => $("#new-config").click());
  $("#export-answers").addEventListener("click", () =>
    download("answers.json", $("#answers-editor").value, "application/json"),
  );
  $("#import-button").addEventListener("click", () =>
    $("#import-file").click(),
  );
  $("#import-file").addEventListener("change", () =>
    action(async () => {
      const file = $("#import-file").files[0];
      if (file) {
        if (file.size > 1048576)
          throw new Error("Answers file must be at most 1 MiB");
        await importAnswers(await file.text());
      }
      $("#import-file").value = "";
    }),
  );
  $("#refresh-jobs").addEventListener("click", () => action(refreshJobs));
  $("#stop-job").addEventListener("click", () =>
    action(async () => {
      if (
        await confirmAction(
          "Stop this job?",
          "Send a graceful stop request through the existing training CLI?",
          "Stop job",
        )
      ) {
        await api("/api/stop", { id: state.selectedJob });
        await refreshJobs();
      }
    }),
  );
  $("#reconcile-job").addEventListener("click", () =>
    action(async () => {
      if (
        await confirmAction(
          "Confirm target cleanup?",
          "Only continue after checking that the previous GPU and Sandbox processes have stopped. This releases the interrupted-job block.",
          "Cleanup confirmed",
        )
      ) {
        await api("/api/reconcile", { id: state.selectedJob, confirmed: true });
        await refreshJobs();
      }
    }),
  );
  $("#copy-log").addEventListener("click", () =>
    download(`${state.selectedJob}.log`, $("#job-log").textContent),
  );
  $("#log-source").addEventListener("change", () => {
    state.logSource = $("#log-source").value;
    action(loadLog);
  });
  setInterval(async () => {
    if (state.busy) return;
    try {
      await refreshJobs();
      $("#connection-text").textContent = "Local session";
      $(".connection").classList.remove("offline");
    } catch {
      $("#connection-text").textContent = "Disconnected";
      $(".connection").classList.add("offline");
    }
  }, 2500);
}
initialize();
