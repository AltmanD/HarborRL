"""Dependency-free tests of consistency evidence and admission decisions."""
import struct

import pytest

from harborrl.consistency.config import require_raw_sampling, validate
from harborrl.consistency.metrics import merge, statistics
from harborrl.consistency.report import assess, decide
from harborrl.consistency.sglang import check_pool, score
from harborrl.consistency.slime import evidence_key


def test_weighted_statistics_ignore_padding_and_tp_duplicates():
    settings = validate({'mode': 'enforce', 'max_abs_delta': 1})
    a = assess([([-1], [-2], [1])], settings)
    b = assess([([-2, -2, -100], [-2, -2, -1], [1, 1, 0])], settings)
    duplicate = {**a, 'contribute': False}
    report = decide([a, b, duplicate], settings, phase='test')
    assert report['status'] == 'passed'
    assert report['metrics']['token_count'] == 3
    assert report['metrics']['mean_abs_delta'] == pytest.approx(1 / 3)
    assert report['metrics']['p99_abs_delta_upper'] == 1


@pytest.mark.parametrize('rows', [[], [([-1], [-1], [0])], [([float('nan')], [-1], [1])],
                                  [([-1], [-1, -2], [1])], [([-1], [-1], [2])]])
def test_structural_errors_fail_audit(rows):
    settings = validate({'mode': 'audit'})
    assert decide([assess(rows, settings)], settings, phase='test')['status'] == 'failed'


def test_thresholds_are_explicit_and_sampling_is_raw():
    with pytest.raises(ValueError):
        validate({'mode': 'enforce'})
    with pytest.raises(ValueError):
        require_raw_sampling({'temperature': .7, 'top_p': 1})
    settings = validate({'mode': 'audit', 'max_abs_delta': .01})
    assert decide([assess([([-1], [-2], [1])], settings)], settings, phase='test')['status'] == 'warning'


def test_serving_evidence_survives_float32_conversion():
    q = [-1.123456789]
    converted = [struct.unpack('!f', struct.pack('!f', q[0]))[0]]
    assert evidence_key([1, 2], q, [1]) == evidence_key([1, 2], converted, [1])
    assert evidence_key([1, 3], q, [1]) != evidence_key([1, 2], converted, [1])


def transport(url, payload=None):
    if payload is None:
        return {'weight_version': '1'}
    assert payload['sampling_params']['max_new_tokens'] == 0
    assert payload['input_ids'] == [10, 11, 12]
    return {'meta_info': {'weight_version': '1', 'input_token_logprobs': [[None, 10], [-1., 11], [-2., 12]]}}


def test_teacher_forcing_exact_response_boundary():
    assert score('http://engine', [10, 11], [12], '1', transport=transport) == [-2.]
    with pytest.raises(ValueError, match='version'):
        score('http://engine', [10, 11], [12], '2', transport=transport)


def test_engine_canary_does_not_dilute_bad_engine():
    settings = validate({'mode': 'enforce', 'max_abs_delta': 2, 'max_mean_abs_delta': .6})
    reference = {'weight_version': '1', 'cases': [{'prompt_ids': [10, 11], 'response_ids': [12], 'logprobs': [-1.]}]}
    states = [{'endpoint': 'http://bad', 'healthy': True, 'weight_version': '1'},
              {'endpoint': 'http://good', 'healthy': True, 'weight_version': '1'}]
    def serving(url, payload=None):
        result = transport(url, payload)
        if payload and 'good' in url:
            result['meta_info']['input_token_logprobs'][-1][0] = -1.
        return result
    report = check_pool(states, reference, settings, transport=serving)
    assert report['metrics']['mean_abs_delta'] == .5
    assert report['status'] == 'failed'


def test_histogram_merge_is_token_weighted():
    parts = [statistics([([-1.] * n, [-1. - delta] * n, [1] * n)]) for n, delta in [(1, .1), (9, .01)]]
    result = merge(parts)
    assert result['mean_abs_delta'] == pytest.approx(.019)
    assert result['p95_abs_delta_upper'] >= .1 - 1e-12
