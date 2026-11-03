# OpenWSP edit format, version 1

An edit is a directory containing `manifest.json` and `edit.npz`. It records tensor replacements against a base model. It contains no full checkpoint. Version 1 preserves the authors' tensor payload convention and adds a version, provenance fields and a content address.

The edit represents **evaluated bfloat16 changes to the listed tensors**. The store retains float32 submissions separately; entries that round back to the base bfloat16 value do not appear in the exported diff.

## Manifest

| Field | Meaning |
| --- | --- |
| `format` | Exactly `openwsp.edit`. |
| `format_version` | Integer `1`. Unknown versions are rejected. |
| `base_model`, `base_revision` | Optional provenance labels, included in the address but not checked against the loaded model. |
| `tensors` | Ordered tensor records; the record's index determines its payload array names. |
| `changed_tensors` | Number of tensor records. |
| `changed_entries_total` | Total bfloat16 entries whose bits changed. |
| `payload_sha256` | Digest of the logical payload arrays, independent of ZIP timestamps or compression. |
| `edit_sha256` | Content address of the canonical manifest, excluding this field itself. |

Each tensor record retains the authors' fields: `name`, `shape`, `dtype`, `form`, `changed_entries`, `base_sha256` and `edited_sha256`. Additional statistics can describe the changed rows, columns and relative change. Tensor hashes are SHA-256 over the tensor's contiguous, C-order, little-endian bfloat16 bit patterns; each element occupies two bytes.

Version 1 checks base compatibility only for tensors listed in the manifest. Callers must separately verify the complete base checkpoint and its pinned revision; unlisted tensors are not checked.

The address commits to the manifest, including provenance and the payload digest. It does not authenticate the author or establish an edit's effect.

## Payload

For a sparse tensor record at index `i`, the NPZ archive contains:

| Array | Type | Meaning |
| --- | --- | --- |
| `t{i}_idx` | Little-endian int32 vector | Flat C-order indices of changed entries. |
| `t{i}_val` | Little-endian uint16 vector | Replacement bfloat16 bit patterns at those indices. |

Indices and values have equal lengths. Import validates names, shapes, array layouts, finite decoded values, changed counts and base/edited hashes before a store applies the edit. Loading does not enable NumPy pickle data.

## Canonical address

Canonical JSON means UTF-8 encoding of `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`.

To calculate `payload_sha256`, visit arrays in ascending name order. For each array, append these bytes to the digest:

1. Length of its canonical JSON header as an unsigned 8-byte little-endian integer.
2. The header, with fields `name`, `dtype` (NumPy's `dtype.str`) and `shape` (an integer list).
3. The array's byte length as an unsigned 8-byte little-endian integer.
4. Its C-order bytes.

Calculate `edit_sha256` over the canonical JSON of the complete manifest with only `edit_sha256` omitted. This includes `payload_sha256`. Repacking the same arrays in another ZIP archive preserves the address; changing a recorded value or manifest field changes it.

## Compatibility with the authors' edits

Unversioned `edit.npz` + `manifest.json` directories from the pinned [authors' release](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math) can be imported. Their base and edited tensor hashes remain mandatory. A legacy artifact has no OpenWSP v1 address until it is exported in v1 format.

Legacy sparse records replace the indexed bfloat16 values. Legacy `lowrank` records also contain `t{i}_a` and `t{i}_b`: start with the base in float64, add each rank's outer product in order, cast to float32, round to bfloat16 using round-to-nearest ties-to-even, then apply indexed sparse fixups. The operation order matters for exact reconstruction. The edited tensor hash must match before the result is accepted.

The public SDK exports with `api.export_edit(path, base_model=..., revision=...)` and imports with `api.apply_edit(path)`. The lower-level store offers `export_edit` and `apply_edit` with the same artifact contract. Protocol refusals still apply when edits are written to a store; format validity is not a bypass for its frozen tensors or other restrictions.
