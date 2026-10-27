"""Terminal interface for the offline catalog, validation and bundle builder."""
import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml

from .catalog import build_catalog
from .presets import PRESETS, worker_defaults
from .validation import read_mapping, validate_paths
from .wizard import build_config_bundle, publish_bundle

DIMENSIONS = {"models": "models", "algorithms": "algorithms", "tasks": "task_providers",
              "harnesses": "harnesses", "datasets": "datasets", "deployments": "deployment_presets",
              "fields": "fields", "compat": "compatibility"}


def ask(label, default=None, convert=str):
    while True:
        value = input(f"{label}" + (f" [{default}]" if default is not None else "") + ": ").strip()
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


def interactive_answers():
    preset = choose("Deployment", [p["id"] for p in PRESETS])
    t, d = {}, {}
    service = preset != "embedded-local"
    print("Components: Qwen / GRPO / Harbor tasks / Claude Code. Advanced fields use an answers file.")
    if service:
        d["manager"] = {
            "source_dir": ask("Controller source directory"),
            "controller_python": ask("Controller Python", sys.executable),
            "control_url": ask("Controller origin (include port 49200)"),
            "gateway_url": ask("Gateway origin (include port 49184)"),
        }
        d["storage"] = {"run_root": ask("Shared run root (absolute)")}
        d["gpu_target"] = {"ssh": {"host": ask("GPU SSH alias/destination")},
                           "source_dir": ask("GPU source directory"), "train_python": ask("GPU training Python")}
    else:
        d["gpu_target"] = {"train_python": ask("Training Python", sys.executable)}
    if preset == "service-ssh":
        workers = []
        count = ask("Number of SSH runners", 1, int)
        if count < 1:
            raise ValueError("At least one SSH runner is required")
        for index in range(count):
            worker = worker_defaults()
            worker.update(id=f"worker-{index}", ssh={"host": ask(f"Runner {index} SSH destination")},
                          source_dir=ask("Runner source directory"), runner_python=ask("Runner Python"),
                          task_roots=[ask("Runner task root")])
            workers.append(worker)
        d["rollout"] = {"workers": workers}
    else:
        t["harbor"] = {"python": ask("Local Harbor runner Python", sys.executable)}
    t["tasks"] = {"catalog": ask("Existing task catalog JSON/JSONL")}
    t["harness"] = {"profile": ask("Existing harness profile JSON")}
    t["model"] = {"checkpoint": ask("GPU checkpoint directory"), "reference": ask("GPU reference directory"),
                  "args_file": ask("Qwen model args preset", "qwen3-8B")}
    t["gateway"] = {"tokenizer_digest": ask("Actual tokenizer SHA-256"),
                     "template_digest": ask("Actual template SHA-256")}
    if not service:
        t["gateway"].update(host=ask("Gateway bind address", "127.0.0.1"),
                            advertised_url=ask("Gateway origin reachable by tasks", "http://127.0.0.1:49184"))
    layout = choose("GPU layout", ["colocate", "split"])
    t["deployment"] = {"layout": layout, "num_gpus": ask("Total GPU budget", 1, int)}
    if layout == "split":
        t["deployment"].update(actor_gpus=ask("Actor GPUs", 1, int), rollout_gpus=ask("Serving GPUs", 1, int))
    t["training"] = {"num_rollout": ask("Rollout batches", 2, int)}
    t["sampling"] = {"group_size": ask("Samples per group", 2, int),
                     "groups_per_batch": ask("Groups per batch", 1, int)}
    answers = {"answers_version": 1, "preset": preset, "training": t, "deployment": d}
    if preset == "service-sandbox-local-runner":
        answers["sandbox_config"] = ask("Existing Sandbox configuration")
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
        if not args.answers and not sys.stdin.isatty():
            parser.error("init requires --answers when stdin is not a terminal")
        output = Path(args.out_dir).expanduser().absolute()
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output directory already exists: {output}")
        if args.probe_local:
            from .probe import probe_local
            print(json.dumps(probe_local(output), indent=2))
        answers = read_mapping(args.answers) if args.answers else interactive_answers()
        bundle = build_config_bundle(answers, output_directory=output)
        render_validation(bundle["validation"])
        if not args.answers:
            print(bundle["files"]["config.yaml"])
            print(bundle["files"]["deployment.yaml"])
            if ask("Publish these files? (yes/no)", "no").lower() != "yes":
                print("Cancelled; no files written")
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
