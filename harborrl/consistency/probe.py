"""Replay a Megatron fixed-token reference against an explicit engine pool."""
import argparse
import json

from harborrl.trajectories.native import digest, publish, read_json

from .config import validate
from .sglang import check_pool, http_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--engine', action='append', required=True)
    parser.add_argument('--config', required=True, help='JSON consistency mapping with explicit tolerances')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    settings = validate(read_json(args.config))
    if settings['mode'] == 'off':
        parser.error('probe mode must be audit or enforce')
    reference = read_json(args.reference)
    if reference.get('digest') != digest({k: v for k, v in reference.items() if k != 'digest'}):
        raise ValueError('Corrupt sync reference')
    states = [{'endpoint': endpoint.rstrip('/'), 'healthy': True,
               'weight_version': http_json(endpoint.rstrip('/') + '/get_weight_version')['weight_version']}
              for endpoint in args.engine]
    report = check_pool(states, reference, settings)
    report['reference_digest'] = reference['digest']
    publish(args.output, report)
    print(json.dumps(report, indent=2))
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
