import json

from albion_bot import config, paths


def test_defaults_when_missing():
    assert config.load() == config.Settings()


def test_round_trip():
    settings = config.Settings(targets=["ore"], route="north", max_minutes=30)
    config.save(settings)
    assert config.load() == settings


def test_bad_values_are_reset():
    paths.SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    paths.SETTINGS.write_text(json.dumps({"confidence": 7, "minimap": [0.9, 0.9, 0.5, 0.5], "route_mode": "fly",
                                          "targets": ["tree"], "unknown": 1}))
    settings = config.load()

    assert settings.confidence == config.Settings().confidence
    assert settings.minimap == config.Settings().minimap
    assert settings.route_mode == "loop"
    assert settings.targets == ["tree"]


def test_broken_file_gives_defaults():
    paths.SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    paths.SETTINGS.write_text("{not json")
    assert config.load() == config.Settings()
