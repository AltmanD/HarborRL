"""Terminal interface for the offline catalog, validation and bundle builder."""
import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml

from .catalog import build_catalog
from .presets import PRESETS, worker_defaults
from .onboarding import load_draft, reference_examples, save_draft
from .validation import read_mapping, validate_paths
from .wizard import build_config_bundle, publish_bundle

DIMENSIONS = {"models": "models", "algorithms": "algorithms", "tasks": "task_providers",
              "harnesses": "harnesses", "datasets": "datasets", "deployments": "deployment_presets",
              "fields": "fields", "compat": "compatibility"}


def ask(label, default=None, convert=str):
    while True:
        value = input(f"{label}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if value == ":save":
            raise SaveDraft
        if not value and default is not None:
            return default
        try:
            if not value:
                raise ValueError("A value is required")
            return convert(value)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)


def choose(label, choices):
    for index, value in enumerate(choices, 1):
        print(f"  {index}. {value}")
    while True:
        index = ask(label, 1, int)
        if 1 <= index <= len(choices):
            return choices[index - 1]
        print("Choose a listed number", file=sys.stderr)


class SaveDraft(Exception):
    pass


def interactive_answers(answers=None, *, output_directory=".", on_change=None):
    from copy import deepcopy

    from harborrl.config import ROOT

    from .onboarding import describe_field
    from .presets import defaults

    answers = deepcopy(answers) if answers else {"answers_version": 1,
        "preset": choose("Deployment", [p["id"] for p in PRESETS]), "training": {}, "deployment": {}}
    preset = answers["preset"]
    training, deployment = defaults(preset, output_directory)
    base = {"training": training, "deployment": deployment,
            "local_environment": {"source_dir": str(ROOT), "python": sys.executable}}
    print("Type ? for help, :skip to leave a field empty, :save to save and exit.")
    print("Progress is saved after every answer. Ctrl+C keeps your draft. New here? Run harborrl hello first.")
    if on_change:
        on_change(answers)

    def get(mapping, path):
        for key in path.split("."):
            mapping = (mapping.get(key, {}) if isinstance(mapping, dict) else
                       mapping[int(key)] if isinstance(mapping, list) and key.isdigit() and int(key) < len(mapping) else {})
        return mapping if mapping != {} else None

    def set_value(path, value):
        parent = answers
        keys = path.split(".")
        for key in keys[:-1]:
            parent = parent[int(key)] if isinstance(parent, list) else parent.setdefault(key, {})
        parent[int(keys[-1]) if isinstance(parent, list) else keys[-1]] = value
        if on_change:
            on_change(answers)

    def prompt(path, fallback=None, convert=str, label=None):
        info = describe_field(path)
        value = get(answers, path)
        if value is None:
            value = fallback if fallback is not None else get(base, path)
        print("  " + info["help"])
        while True:
            text = input(f"{label or info['label']}" + (f" [{value}]" if value is not None else " [required]") + ": ").strip()
            if text == "?":
                print(info["help"])
                continue
            if text == ":save":
                raise SaveDraft
            try:
                selected = None if text == ":skip" else (value if not text else convert(text))
                if selected is None and text != ":skip":
                    raise ValueError("A value is required; use :skip to continue in a draft.")
            except ValueError as exc:
                print(exc, file=sys.stderr)
                continue
            set_value(path, selected)
            return selected

    if preset == "embedded-local":
        prompt("local_environment.source_dir")
        prompt("local_environment.python")
    else:
        for path in ("deployment.manager.source_dir", "deployment.manager.controller_python",
                     "deployment.manager.control_url", "deployment.manager.gateway_url",
                     "deployment.gpu_target.ssh.host", "deployment.gpu_target.source_dir",
                     "deployment.gpu_target.train_python"):
            prompt(path)
        if preset == "service-ssh":
            workers = get(answers, "deployment.rollout.workers") or []
            count = ask("SSH runner count (each needs Harbor, Docker and task files; :save to exit)", len(workers) or 1, int)
            if count < 1:
                raise ValueError("At least one SSH runner is required")
            workers = workers[:count] + [{**worker_defaults(), "id": f"worker-{index}"}
                                        for index in range(len(workers), count)]
            set_value("deployment.rollout.workers", workers)
            for index in range(count):
                for key in ("id", "ssh.host", "source_dir", "runner_python", "task_roots.0"):
                    prompt(f"deployment.rollout.workers.{index}.{key}")
        else:
            prompt("training.harbor.python", sys.executable)
            prompt("sandbox_config")
    prompt("deployment.storage.run_root")
    for path in ("training.tasks.catalog", "training.harness.profile", "training.model.checkpoint",
                 "training.model.reference", "training.model.args_file"):
        prompt(path)
    if preset == "embedded-local":
        prompt("training.gateway.host")
        prompt("training.gateway.advertised_url")
    prompt("training.deployment.layout", label="GPU layout (colocate or split)")
    prompt("training.deployment.num_gpus", convert=int, label="Total GPU budget (check model memory first)")
    if get(answers, "training.deployment.layout") == "split":
        prompt("training.deployment.actor_gpus", convert=int)
        prompt("training.deployment.rollout_gpus", convert=int)
    prompt("training.training.num_rollout", convert=int, label="Number of rollout batches")
    prompt("training.sampling.group_size", convert=int, label="Samples per prompt (GRPO requires at least 2)")
    prompt("training.sampling.groups_per_batch", convert=int, label="Complete prompt groups per batch")
    return answers


def render_validation(result):
    print(f"{result['status']} (scope: {result['scope']})")
    for row in result["checks"]:
        print(f"  {row['status']}: {row['path']}: {row['message']}")


def render_catalog(result):
    print("HarborRL options (catalog 1; training schema 2; deployment schema 1)")
    for name, value in result.items():
        if name in {"catalog_version", "schema_versions"}:
            continue
        if name == "fields" and len(result) > 3:
            print("fields: use 'harborrl options fields' for field descriptions")
        elif isinstance(value, list):
            print(f"{name}:")
            for row in value:
                description = row.get("description", row.get("display_name", ""))
                if "available" in row:
                    description += f" (status={row['status']}, locally_available={row['available']})"
                print(f"  {row.get('id', row.get('name'))}: {description}")
        else:
            print(yaml.safe_dump({name: value}, sort_keys=False).rstrip())


def main(argv):
    parser = argparse.ArgumentParser(prog="harborrl")
    commands = parser.add_subparsers(dest="command", required=True)
    options = commands.add_parser("options", help="List supported configuration options")
    options.add_argument("dimension", nargs="?", choices=DIMENSIONS)
    options.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate", help="Validate offline; no runtime is started")
    validate.add_argument("config")
    validate.add_argument("--deployment")
    validate.add_argument("--check-files", action="store_true")
    validate.add_argument("--json", action="store_true")
    init = commands.add_parser("init", help="Create a configuration bundle in a new directory")
    init.add_argument("--out-dir", required=True)
    init.add_argument("--answers")
    init.add_argument("--probe-local", action="store_true")
    init.add_argument("--resume", help="Resume a saved draft JSON")
    init.add_argument("--draft-file", help="Progress file (default: <out-dir>.draft.json)")
    init.add_argument("--save-draft", action="store_true", help="Save incomplete answers without publishing a runnable bundle")
    init.add_argument("--example", choices=["local", "sandbox"], help="Save a filled reference draft; replace site paths before use")
    init.add_argument("--prepare-model", action="store_true", help="Recalculate fingerprints using the target Python")
    args = parser.parse_args(argv)
    try:
        if args.command == "options":
            result = build_catalog()
            if args.dimension:
                key = DIMENSIONS[args.dimension]
                result = {k: result[k] for k in ("catalog_version", "schema_versions", key)}
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                render_catalog(result)
            return 0
        if args.command == "validate":
            result = validate_paths(args.config, args.deployment, check_files=args.check_files)
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                render_validation(result)
            return int(result["status"] == "invalid")
        if not (args.answers or args.resume or args.example or args.save_draft) and not sys.stdin.isatty():
            parser.error("init requires --answers when stdin is not a terminal")
        output = Path(args.out_dir).expanduser().absolute()
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output directory already exists: {output}")
        if args.probe_local:
            from .probe import probe_local
            print(json.dumps(probe_local(output), indent=2))
        if args.answers and args.resume:
            raise ValueError("Choose --answers or --resume")
        draft_path = Path(args.draft_file or args.resume or (str(output) + ".draft.json")).expanduser().absolute()
        revision = load_draft(draft_path)["revision"] if args.resume and draft_path.exists() else None
        if not args.resume and (draft_path.exists() or draft_path.is_symlink()):
            raise ValueError(f"Draft already exists: {draft_path}. Use --resume or choose a new --draft-file.")

        def persist(current):
            nonlocal revision
            revision = save_draft(draft_path, current, revision=revision)["revision"]

        answers = load_draft(args.resume)["answers"] if args.resume else (read_mapping(args.answers) if args.answers else None)
        if args.example:
            answers = reference_examples(output)[0 if args.example == "local" else 1]["answers"]
        if args.save_draft or args.example:
            persist(answers or {"answers_version": 1, "preset": "embedded-local"})
            print(f"Draft saved: {draft_path}\nResume: harborrl init --out-dir {shlex.quote(str(output))} --resume {shlex.quote(str(draft_path))}")
            return 0
        interactive = not args.answers and sys.stdin.isatty()
        if interactive:
            try:
                answers = interactive_answers(answers, output_directory=output, on_change=persist)
            except (SaveDraft, EOFError, KeyboardInterrupt):
                if draft_path.is_file():
                    print(f"Draft kept at {draft_path}; resume with --resume {shlex.quote(str(draft_path))}")
                else:
                    print("Cancelled before a deployment was selected; no draft created")
                return 0
        if answers is None:
            raise ValueError("Supply --answers or --resume")
        if interactive or args.prepare_model or not all(answers.get("training", {}).get("gateway", {}).get(key)
                                                        for key in ("tokenizer_digest", "template_digest")):
            from .fingerprints import prepare_model
            print("Preparing model fingerprints automatically with the GPU target tokenizer...")
            try:
                prepared = prepare_model(answers, output)
                answers.setdefault("training", {}).setdefault("gateway", {}).update(prepared["fingerprints"])
            except ValueError:
                if revision is None or interactive:
                    persist(answers)
                print(f"Draft kept at {draft_path}. Fix the environment/checkpoint, then resume.", file=sys.stderr)
                raise
        if interactive:
            persist(answers)
        bundle = build_config_bundle(answers, output_directory=output)
        render_validation(bundle["validation"])
        if interactive:
            print(bundle["files"]["config.yaml"])
            print(bundle["files"]["deployment.yaml"])
            try:
                publish = ask("Publish these files? (yes/no)", "no").lower() == "yes"
            except (SaveDraft, EOFError, KeyboardInterrupt):
                publish = False
            if not publish:
                print(f"Draft saved: {draft_path}; no runnable bundle published")
                return 0
        publish_bundle(bundle, output)
        print(f"Created configuration bundle: {output}")
        config = shlex.quote(str(output / "config.yaml"))
        deployment = shlex.quote(str(output / "deployment.yaml"))
        for command, suffix in (("validate", " --check-files"), ("train", " --dry-run"), ("doctor", "")):
            print(f"harborrl {command} {config} --deployment {deployment}{suffix}")
        print("doctor checks the live environment and connectivity; training requires a separate command.")
        return 0
    except (ValueError, TypeError, KeyError, AttributeError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("Cancelled; no configuration published", file=sys.stderr)
        return 130
