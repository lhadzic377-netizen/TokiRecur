"""Unit tests for selectable adapter modes.

An adapter is not a model: it is a second GGUF that changes how a base
model runs. The registry keeps a catalog per model and one active id.
Three things have to hold or "plug and play" is a lie:

1. the old flat ``draft:`` layout still resolves to a mode;
2. selecting a mode rewrites the catalog, the active id and the flat
   projection in one write, so launch and fit read one thing;
3. an adapter that cannot be attached (missing file, stale id, unknown
   kind, build without DSpine) is refused before the launch, not deep
   in the model load.
"""

from pathlib import Path

import pytest
import yaml

from tokirecur import adapters as adapters_mod
from tokirecur.adapters import (
    BASE_MODE_ID,
    KIND_DRAFT,
    KIND_LORA,
    AdapterMode,
    active_mode,
    active_mode_id,
    adapter_launch_args,
    adapter_modes,
    available_modes,
    choice_map,
    describe_active,
    discover_adapters,
    kind_for_path,
    mode_fields,
    mode_id_for,
    resolve_draft,
)
from tokirecur.ctl import (
    _build_spec,
    _check_draft,
    _llama_args,
    install_adapter,
    load_registry,
    pull_model,
    remove_adapter,
    save_registry,
    update_spec_field,
)
from tokirecur.draftmatch import adapter_target_family


def _spec(tmp_path: Path, name: str = "base", **extra):
    model_dir = tmp_path / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "Qwen3.8-27B-Q4_K_M.gguf"
    target.write_bytes(b"x")
    raw = {
        "name": name,
        "label": "Qwen3.8-27B",
        "gguf": str(target),
        "llama_port": 18080,
        "daemon_port": 18081,
        "ctx_size": 4096,
        "parallel": 1,
        "kv": "f16",
        "max_tokens": 1024,
        "model_id": name,
    }
    raw.update(extra)
    return _build_spec(raw)


def _mode(tmp_path: Path, *, mode_id: str = "dspine-q4", kind: str = KIND_DRAFT,
          n_max: int = 9, extra_args: tuple[str, ...] = ()) -> AdapterMode:
    path = tmp_path / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    path.write_bytes(b"x")
    return AdapterMode(
        id=mode_id,
        label="DSpine Q4_K_M",
        kind=kind,
        gguf=str(path),
        n_max=n_max,
        repo="aj9o9/Qwen3.8-27B-DSpine-b16-GGUF",
        file=path.name,
        extra_args=extra_args,
    )


def _save(specs, reg: Path) -> None:
    save_registry(list(specs), reg)


# ── the mode object ────────────────────────────────────────────────────


def test_a_draft_mode_round_trips_and_emits_its_own_argv(tmp_path):
    mode = _mode(tmp_path, extra_args=("--spec-draft-type-k", "q8_0"))
    restored = AdapterMode.from_dict(
        yaml.safe_load(yaml.safe_dump(mode.to_dict()))
    )
    assert restored == mode
    args = restored.launch_args()
    assert args[:2] == ["--spec-draft-model", str(mode.path)]
    assert args[args.index("--spec-type") + 1] == "draft-dflash"
    assert args[-2:] == ["--spec-draft-type-k", "q8_0"]


def test_a_lora_mode_uses_lora_not_speculative_flags(tmp_path):
    mode = _mode(tmp_path, mode_id="lora-style", kind=KIND_LORA)
    assert mode.launch_args() == ["--lora", str(mode.path)]
    # A user-supplied --lora-scaled is not duplicated by the mode.
    scaled = AdapterMode(
        id="scaled", kind=KIND_LORA, gguf=mode.gguf,
        extra_args=("--lora-scaled", mode.gguf, "0.5"),
    )
    assert scaled.launch_args() == ["--lora-scaled", mode.gguf, "0.5"]


def test_an_unknown_kind_is_kept_but_refuses_at_launch(tmp_path):
    mode = _mode(tmp_path, mode_id="mystery", kind="quantum")
    args = mode.launch_args
    with pytest.raises(ValueError, match="quantum"):
        args()


# ── legacy and catalog resolution ──────────────────────────────────────


def test_a_legacy_flat_draft_is_still_an_active_mode(tmp_path):
    mode = _mode(tmp_path)
    spec = _spec(tmp_path, draft=mode.gguf, draft_n_max=7,
                 draft_spec_type="draft-dflash", draft_repo=mode.repo)
    assert adapter_modes(spec) == []
    assert [m.id for m in available_modes(spec)] == [mode_id_for(mode.gguf)]
    active = active_mode(spec)
    assert active is not None and active.gguf == mode.gguf
    assert resolve_draft(spec) is not None
    assert resolve_draft(spec).n_max == 7


def test_a_catalog_without_an_active_id_runs_the_base(tmp_path):
    mode = _mode(tmp_path)
    spec = _spec(tmp_path, adapters=[mode.to_dict()], adapter_mode="")
    assert active_mode(spec) is None
    assert resolve_draft(spec) is None
    assert describe_active(spec) == "base model (no adapter)"
    choices = choice_map(spec)
    assert choices["base model (no adapter)"] == BASE_MODE_ID
    assert choices["DSpine Q4_K_M"] == mode.id


def test_selecting_a_mode_rewrites_the_whole_projection(tmp_path):
    mode = _mode(tmp_path)
    spec = _spec(tmp_path, adapters=[mode.to_dict()], adapter_mode="")
    fields = mode_fields(spec, mode.id)
    assert fields["adapter_mode"] == mode.id
    assert fields["draft"] == mode.gguf
    assert fields["draft_n_max"] == mode.n_max
    assert fields["adapters"] == [mode.to_dict()]
    # And back to the base: catalog stays, projection clears.
    base = mode_fields(spec, BASE_MODE_ID)
    assert base["adapter_mode"] == ""
    assert base["draft"] == ""
    assert base["draft_n_max"] == 0
    assert base["adapters"] == [mode.to_dict()]


def test_an_unknown_mode_id_is_refused(tmp_path):
    spec = _spec(tmp_path, adapters=[_mode(tmp_path).to_dict()])
    with pytest.raises(KeyError, match="not installed"):
        mode_fields(spec, "no-such-mode")


def test_a_lora_mode_projects_no_draft_fields(tmp_path):
    mode = _mode(tmp_path, mode_id="lora", kind=KIND_LORA)
    spec = _spec(tmp_path, adapters=[mode.to_dict()])
    fields = mode_fields(spec, mode.id)
    assert fields["adapter_mode"] == mode.id
    assert fields["draft"] == ""
    assert active_mode(_build_spec({**spec.raw, **fields})) is not None
    assert adapter_launch_args(_build_spec({**spec.raw, **fields})) == [
        "--lora", mode.gguf
    ]


# ── discovery: an adapter dropped beside the model ─────────────────────


def test_discovery_offers_an_unregistered_adapter_and_skips_the_model(
        tmp_path):
    spec = _spec(tmp_path)
    model_dir = Path(spec.gguf).parent
    adapter = model_dir / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    adapter.write_bytes(b"x")
    (model_dir / "mmproj-F16.gguf").write_bytes(b"x")
    (model_dir / "notes.gguf").write_bytes(b"x")
    found = discover_adapters(spec)
    assert [Path(m.gguf).name for m in found] == [adapter.name]
    assert found[0].id == mode_id_for(adapter.name)
    assert "found on disk" in " ".join(choice_map(spec))


def test_discovery_finds_an_adapter_in_its_own_sibling_directory(tmp_path):
    spec = _spec(tmp_path)
    models = Path(spec.gguf).parent.parent
    sibling = models / "qwen3.8-27b-dspine-b16-gguf"
    sibling.mkdir()
    adapter = sibling / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    adapter.write_bytes(b"x")
    unrelated = models / "laguna-dspine-b16-gguf"
    unrelated.mkdir()
    (unrelated / "Laguna-DSpine-b16-Q4_K_M.gguf").write_bytes(b"x")

    found = discover_adapters(spec)
    assert [Path(m.gguf).parent for m in found] == [sibling]


def test_discovery_does_not_re_offer_a_registered_mode(tmp_path):
    spec = _spec(tmp_path)
    adapter = Path(spec.gguf).parent / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    adapter.write_bytes(b"x")
    mode = AdapterMode(id=mode_id_for(adapter.name), gguf=str(adapter))
    spec = _build_spec({**spec.raw, "adapters": [mode.to_dict()]})
    assert discover_adapters(spec) == []


def test_kind_for_path_reads_dspine_by_name(tmp_path):
    path = tmp_path / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    path.write_bytes(b"x")
    assert kind_for_path(path) == KIND_DRAFT
    assert kind_for_path(tmp_path / "target.gguf") is None


def test_selecting_a_legacy_mode_migrates_it_into_the_catalog(tmp_path):
    mode = _mode(tmp_path)
    spec = _spec(tmp_path, draft=mode.gguf, draft_n_max=7)
    fields = mode_fields(spec, mode_id_for(mode.gguf))
    assert fields["adapter_mode"] == mode_id_for(mode.gguf)
    assert fields["draft"] == mode.gguf
    assert fields["draft_n_max"] == 7
    assert [m["gguf"] for m in fields["adapters"]] == [mode.gguf]


def test_selecting_a_discovered_mode_registers_it(tmp_path):
    spec = _spec(tmp_path)
    adapter = Path(spec.gguf).parent / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    adapter.write_bytes(b"x")
    fields = mode_fields(spec, mode_id_for(adapter.name))
    assert fields["adapter_mode"] == mode_id_for(adapter.name)
    assert [m["gguf"] for m in fields["adapters"]] == [str(adapter)]
    reloaded = _build_spec({**spec.raw, **fields})
    assert active_mode(reloaded).gguf == str(adapter)


# ── registry edits and install ─────────────────────────────────────────


def test_update_spec_field_switches_modes_and_returns_to_base(tmp_path):
    reg = tmp_path / "models.yaml"
    mode = _mode(tmp_path)
    spec = _spec(tmp_path, adapters=[mode.to_dict()], adapter_mode="")
    _save([spec], reg)

    updated = update_spec_field("base", "adapter_mode", mode.id, registry=reg)
    assert updated.adapter_mode == mode.id
    assert updated.draft == mode.gguf
    on_disk = load_registry(reg)[0]
    assert on_disk.adapter_mode == mode.id
    assert on_disk.draft_n_max == mode.n_max

    back = update_spec_field(
        "base", "adapter_mode", BASE_MODE_ID, registry=reg,
    )
    assert back.adapter_mode == ""
    assert back.draft == ""
    assert adapter_modes(back) == [mode]


def test_update_spec_field_refuses_an_unknown_mode(tmp_path):
    reg = tmp_path / "models.yaml"
    _save([_spec(tmp_path, adapters=[_mode(tmp_path).to_dict()])], reg)
    with pytest.raises(KeyError, match="not installed"):
        update_spec_field("base", "adapter_mode", "nope", registry=reg)


def test_applying_an_unloadable_mode_is_refused_before_the_write(
        tmp_path, monkeypatch):
    reg = tmp_path / "models.yaml"
    mode = _mode(tmp_path)
    _save([_spec(tmp_path, adapters=[mode.to_dict()], adapter_mode="")], reg)
    monkeypatch.setattr("tokirecur.ctl.draft_needs_dspine", lambda draft: True)
    monkeypatch.setattr("tokirecur.ctl.draft_supported_by", lambda binary: False)
    with pytest.raises(ValueError, match="cannot load"):
        update_spec_field(
            "base", "adapter_mode", mode.id, registry=reg, apply=True,
        )
    # The registry still runs the base: a refused switch writes nothing.
    assert load_registry(reg)[0].adapter_mode == ""
    assert load_registry(reg)[0].draft == ""


def test_install_adapter_from_a_file_registers_and_activates(tmp_path):
    reg = tmp_path / "models.yaml"
    _save([_spec(tmp_path)], reg)
    source = tmp_path / "elsewhere" / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    source.parent.mkdir()
    source.write_bytes(b"x")

    spec, mode, active = install_adapter(
        str(source), model="base", registry=reg,
    )
    assert active is True
    assert spec.adapter_mode == mode.id
    target_dir = Path(spec.gguf).parent
    assert Path(mode.gguf).parent == target_dir.resolve()
    assert Path(mode.gguf).exists()
    assert spec.draft == mode.gguf
    assert mode.kind == KIND_DRAFT
    assert mode in adapter_modes(spec)

    # Re-installing the same adapter does not duplicate the mode.
    again, same, _ = install_adapter(str(source), model="base", registry=reg)
    assert same.id == mode.id
    assert [m.id for m in adapter_modes(again)] == [mode.id]


def test_installing_on_a_legacy_model_keeps_its_draft_active(tmp_path):
    reg = tmp_path / "models.yaml"
    first = _mode(tmp_path)
    _save([_spec(tmp_path, draft=first.gguf, draft_n_max=7)], reg)
    second = tmp_path / "Qwen3.8-27B-DSpine-b16-Q8_0.gguf"
    second.write_bytes(b"x")

    updated, mode, active = install_adapter(
        str(second), model="base", registry=reg,
    )
    assert active is False
    assert updated.adapter_mode == mode_id_for(first.gguf)
    assert updated.draft == first.gguf
    assert {m.id for m in adapter_modes(updated)} == {
        mode_id_for(first.gguf), mode_id_for(second.name),
    }


def test_install_adapter_uses_a_discoverable_sibling_in_place(tmp_path):
    reg = tmp_path / "models.yaml"
    base = _spec(tmp_path)
    _save([base], reg)
    sibling = Path(base.gguf).parent.parent / "qwen3.8-27b-dspine-b16-gguf"
    sibling.mkdir()
    adapter = sibling / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    adapter.write_bytes(b"x")

    updated, mode, _ = install_adapter(
        str(adapter), model="base", registry=reg,
    )
    assert Path(mode.gguf) == adapter
    assert updated.draft == str(adapter)
    # No duplicate copy in the target directory.
    assert not (Path(base.gguf).parent / adapter.name).exists()


def test_install_adapter_can_stay_inactive(tmp_path):
    reg = tmp_path / "models.yaml"
    _save([_spec(tmp_path)], reg)
    source = tmp_path / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    source.write_bytes(b"x")
    spec, mode, active = install_adapter(
        str(source), model="base", activate=False, registry=reg,
    )
    assert active is False
    assert spec.adapter_mode == ""
    assert spec.draft == ""
    assert adapter_modes(spec) == [mode]


def test_install_adapter_infers_the_target_from_the_file_family(tmp_path):
    reg = tmp_path / "models.yaml"
    _save([_spec(tmp_path)], reg)
    source = Path(_spec(tmp_path, "other").gguf).parent / (
        "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    )
    # The file is not in the base's directory and no --model is given:
    # the family name still finds the one installed match.
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"x")
    spec, mode, _ = install_adapter(str(source), registry=reg)
    assert spec.name == "base"


def test_an_ambiguous_adapter_target_is_refused(tmp_path):
    reg = tmp_path / "models.yaml"
    _save([_spec(tmp_path, "base-a"), _spec(tmp_path, "base-b")], reg)
    source = tmp_path / "loose" / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    source.parent.mkdir()
    source.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="several installed models"):
        install_adapter(str(source), registry=reg)


def test_install_adapter_refuses_a_file_that_is_not_an_adapter(tmp_path):
    reg = tmp_path / "models.yaml"
    _save([_spec(tmp_path)], reg)
    plain = tmp_path / "plain.gguf"
    plain.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="does not look like an adapter"):
        install_adapter(str(plain), model="base", registry=reg)


def test_pulling_an_adapter_repo_installs_it_on_the_installed_target(
        tmp_path, monkeypatch):
    reg = tmp_path / "models.yaml"
    base = _spec(tmp_path, "qwen3.8-27b-gguf")
    _save([base], reg)
    fetched = Path(base.gguf).parent / "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    fetched.write_bytes(b"x")

    class FakeApi:
        def __init__(self, endpoint=None):
            pass

        def list_repo_files(self, _repo):
            return [fetched.name, "README.md"]

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: str(fetched),
    )
    updated = pull_model(
        "aj9o9/Qwen3.8-27B-DSpine-b16-GGUF", registry=reg,
    )
    assert updated.name == base.name
    assert updated.adapter_mode == mode_id_for(fetched.name)
    assert updated.draft == str(fetched)
    assert [m["gguf"] for m in updated.raw["adapters"]] == [str(fetched)]


def test_remove_adapter_switches_an_active_model_to_base(tmp_path):
    reg = tmp_path / "models.yaml"
    mode = _mode(tmp_path)
    spec = _spec(tmp_path)
    _save([spec], reg)
    install_adapter(mode.gguf, model="base", registry=reg)
    active_spec = load_registry(reg)[0]
    assert active_mode_id(active_spec)

    removed, was_active = remove_adapter("base", active_mode_id(active_spec),
                                         registry=reg)
    assert was_active is True
    assert removed.adapter_mode == ""
    assert removed.draft == ""
    assert adapter_modes(removed) == []


# ── launch gate ────────────────────────────────────────────────────────


def test_check_draft_refuses_a_stale_mode_id(tmp_path):
    spec = _spec(tmp_path, adapter_mode="missing-mode")
    with pytest.raises(RuntimeError, match="not in .*adapter catalog"):
        _check_draft(spec)


def test_check_draft_refuses_a_missing_adapter_file(tmp_path):
    mode = _mode(tmp_path)
    Path(mode.gguf).unlink()
    spec = _spec(tmp_path, adapters=[mode.to_dict()], adapter_mode=mode.id)
    with pytest.raises(RuntimeError, match="is missing"):
        _check_draft(spec)


def test_check_draft_refuses_an_unknown_kind(tmp_path):
    mode = _mode(tmp_path, mode_id="mystery", kind="quantum")
    spec = _spec(tmp_path, adapters=[mode.to_dict()], adapter_mode=mode.id)
    with pytest.raises(RuntimeError, match="quantum"):
        _check_draft(spec)


def test_llama_args_carry_a_lora_mode(tmp_path):
    mode = _mode(tmp_path, mode_id="lora", kind=KIND_LORA)
    spec = _spec(tmp_path, adapters=[mode.to_dict()], adapter_mode=mode.id)
    args = _llama_args(spec)
    assert args[args.index("--lora") + 1] == mode.gguf


# ── family matching ────────────────────────────────────────────────────


@pytest.mark.parametrize("name,family", [
    ("aj9o9/Qwen3.8-27B-DSpine-b16-GGUF", "Qwen3.8-27B"),
    ("Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf", "Qwen3.8-27B"),
    ("Qwen3.8-27B-DSpine-GGUF", "Qwen3.8-27B"),
    ("Qwen3.8-27B-GGUF", "Qwen3.8-27B"),
])
def test_adapter_target_family(name, family):
    assert adapter_target_family(name) == family


def test_mode_id_is_stable_and_path_free(tmp_path):
    assert mode_id_for("/a/b/Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf") == \
        mode_id_for("Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf")
    assert mode_id_for("Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf") == \
        "qwen3.8-27b-dspine-b16-q4_k_m"


def test_adapters_module_exports_are_consistent():
    assert set(adapters_mod.ADAPTER_KINDS) == {KIND_DRAFT, KIND_LORA}
