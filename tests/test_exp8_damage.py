"""KL checks use analytic distributions and hash-bound local arrays only."""
import hashlib
import json
import math

import numpy as np
import pytest

from replications.exp8.damage import (
    PRODUCTION_SCOPE,
    kl_from_logits,
    main,
    reduce_kl_chunks,
    reduce_manifest,
    validate_cpu,
)


def test_analytic_two_token_kl_and_direction():
    p = np.array([[0.75, 0.25], [0.2, 0.8]])
    q = np.array([[0.5, 0.5], [0.6, 0.4]])
    expected = sum(float((left * np.log(left / right)).sum()) for left, right in zip(p, q, strict=True))
    measured = kl_from_logits(np.log(p), np.log(q))
    assert measured['token_count'] == 2
    assert measured['kl_sum'] == pytest.approx(expected, abs=1e-14)
    assert measured['kl_mean'] == pytest.approx(expected / 2, abs=1e-14)
    assert measured['kl_sum'] != pytest.approx(kl_from_logits(np.log(q), np.log(p))['kl_sum'])


def test_identical_offset_logits_and_mask_excludes_padding():
    base = np.array([[[1.0, -2.0], [10000.0, -10000.0]], [[0.0, 0.0], [1.0, 2.0]]])
    edited = base.copy()
    edited[0, 1] *= -1
    edited[1, 0] = [9999, -9999]
    mask = np.array([[True, False], [False, True]])
    assert kl_from_logits(base, edited, mask) == {'token_count': 2, 'kl_sum': 0.0, 'kl_mean': 0.0}
    assert kl_from_logits(base, base + 20000)['kl_sum'] == 0.0
    assert kl_from_logits(base, edited)['kl_mean'] > 1000


def test_unequal_chunk_token_counts_use_weighted_mean_and_batch_mask():
    base = np.log(np.array([[0.8, 0.2], [0.5, 0.5], [0.25, 0.75], [0.4, 0.6]]))
    edited = np.log(np.array([[0.2, 0.8], [0.3, 0.7], [0.5, 0.5], [0.45, 0.55]]))
    complete = kl_from_logits(base, edited)
    chunked = reduce_kl_chunks([(base[:1], edited[:1], None), (base[1:], edited[1:], None)])
    assert chunked == pytest.approx(complete)
    unweighted = (kl_from_logits(base[:1], edited[:1])['kl_mean'] + kl_from_logits(base[1:], edited[1:])['kl_mean']) / 2
    assert chunked['kl_mean'] != pytest.approx(unweighted)
    masked = kl_from_logits(base.reshape(2, 2, 2), edited.reshape(2, 2, 2), np.array([[True, False], [True, True]]))
    assert masked == pytest.approx(kl_from_logits(base[[0, 2, 3]], edited[[0, 2, 3]]))


def test_empty_or_all_padding_has_no_defined_mean():
    empty = {'token_count': 0, 'kl_sum': 0.0, 'kl_mean': None}
    assert kl_from_logits(np.zeros((0, 3)), np.zeros((0, 3))) == empty
    assert kl_from_logits(np.ones((2, 3)), np.zeros((2, 3)), np.zeros(2, dtype=bool)) == empty
    assert reduce_kl_chunks(iter([])) == empty


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_logits_rejected_even_in_padding(bad):
    base = np.zeros((2, 2))
    edited = base.copy()
    edited[1, 0] = bad
    with pytest.raises(ValueError, match='finite'):
        kl_from_logits(base, edited, np.array([True, False]))


@pytest.mark.parametrize('mask', [[1, 0], [True], [[True, True]], [0.5, 1.0]])
def test_invalid_mask_is_not_broadcast_or_coerced(mask):
    with pytest.raises(ValueError, match='mask'):
        kl_from_logits(np.zeros((2, 3)), np.zeros((2, 3)), mask)


def test_logits_reject_mismatch_nonreal_and_unrepresentable_values():
    with pytest.raises(ValueError, match='shapes'):
        kl_from_logits(np.zeros((2, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match='vocabulary'):
        kl_from_logits(np.zeros((2, 0)), np.zeros((2, 0)))
    with pytest.raises(TypeError, match='real numeric'):
        kl_from_logits(np.array([[1+2j, 0]]), np.zeros((1, 2)))
    with pytest.raises(ValueError, match='float64'):
        kl_from_logits(np.array([[1e308, -1e308]]), np.zeros((1, 2)))


def make_manifest(tmp_path, *, synthetic=True):
    count = 2 if synthetic else 512
    base = np.zeros((count, 2, 2))
    edited = np.tile(np.log([0.75, 0.25]), (count, 2, 1))
    mask = np.tile(np.array([False, True]), (count, 1))
    def record(name, array):
        path = tmp_path / name
        np.save(path, array)
        return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    examples = [{'id': f'ref-{i}', 'text_sha256': hashlib.sha256(f'reference-{i}'.encode()).hexdigest(),
                 'source': 'gsm8k' if i < 256 else 'math'} for i in range(count)]
    manifest = {'schema_version': 1, 'synthetic': synthetic, 'examples': examples,
                'chunks': [{'base_logits': record('base.npy', base),
                            'edited_logits': record('edited.npy', edited),
                            'valid_mask': record('mask.npy', mask),
                            'reference_ids': [row['id'] for row in examples]}]}
    if not synthetic:
        manifest.update(scope=PRODUCTION_SCOPE, base_checkpoint={'sha256': 'a'*64},
                        edited_checkpoint={'sha256': 'b'*64}, edit_sha256='c'*64)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_manifest_reduction_preserves_identities_not_logits_or_text(tmp_path):
    path, manifest = make_manifest(tmp_path)
    result = reduce_manifest(path)
    assert result['token_count'] == 2
    assert result['kl_mean'] == pytest.approx(0.5 * math.log(4/3))
    assert result['provenance']['synthetic'] is True
    assert result['provenance']['reference_identities'] == manifest['examples']
    assert result['provenance']['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert 'logits' not in result and 'texts' not in result
    assert 'no model' in result['provenance']['scope'] or 'not a model' in result['provenance']['scope']


def test_production_manifest_requires_full_balanced_scope_and_preserves_edit(tmp_path):
    path, manifest = make_manifest(tmp_path, synthetic=False)
    result = reduce_manifest(path)
    assert result['token_count'] == 512
    assert result['provenance']['edit_sha256'] == 'c'*64
    assert result['provenance']['synthetic'] is False
    manifest['examples'] = manifest['examples'][:-1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='512'):
        reduce_manifest(path)


@pytest.mark.parametrize('change,expected', [
    ('hash', 'SHA-256'), ('escape', 'within'), ('duplicate', 'unique'),
    ('missing_coverage', 'Every declared'), ('no_valid_tokens', 'valid continuation'),
    ('pickle', 'Object arrays'), ('undeclared', 'declared example'),
])
def test_manifest_rejects_invalid_evidence(tmp_path, change, expected):
    path, manifest = make_manifest(tmp_path)
    chunk = manifest['chunks'][0]
    if change == 'hash':
        chunk['base_logits']['sha256'] = '0'*64
    elif change == 'escape':
        chunk['base_logits']['path'] = '../base.npy'
    elif change == 'duplicate':
        manifest['examples'][1]['id'] = manifest['examples'][0]['id']
    elif change == 'missing_coverage':
        chunk['reference_ids'] = ['ref-0', 'ref-0']
    elif change == 'undeclared':
        chunk['reference_ids'][1] = 'unknown'
    elif change in ('no_valid_tokens', 'pickle'):
        key = 'valid_mask' if change == 'no_valid_tokens' else 'base_logits'
        array_path = tmp_path / chunk[key]['path']
        np.save(array_path, np.zeros((2, 2), dtype=bool) if change == 'no_valid_tokens' else np.array([object()]))
        chunk[key]['sha256'] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=expected):
        reduce_manifest(path)


def test_manifest_rejects_symlink_escape_and_preserves_existing_output(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    path, manifest = make_manifest(data)
    target = tmp_path / 'outside.npy'
    np.save(target, np.zeros((2, 2, 2)))
    (data/'base.npy').unlink()
    (data/'base.npy').symlink_to(target)
    manifest['chunks'][0]['base_logits']['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    output = data/'summary.json'
    output.write_text('previous evidence')
    with pytest.raises(ValueError, match='escapes'):
        main(['reduce', '--manifest', str(path), '--output', str(output)])
    assert output.read_text() == 'previous evidence'


def test_cpu_fixture_cli_is_explicitly_synthetic_and_reportable(tmp_path):
    from openwsp.reporting import generate_report

    results = tmp_path/'replications/exp8/results'
    path = results/'damage_cpu_validation/summary.json'
    main(['validate', '--output', str(path)])
    result = json.loads(path.read_text())
    assert result == validate_cpu()
    assert all(result['checks'].values())
    assert result['provenance']['synthetic'] is True
    assert result['model_measurement'] == {'status': 'pending', 'kl_mean': None}
    assert 'accuracy' not in result
    generate_report(tmp_path, results)
    report = (tmp_path/'README.md').read_text()
    assert 'damage_cpu_validation' in report and 'CPU analysis and implementation checks' in report
    assert 'not GPU benchmark results' in report
