"""
Tests for the settings file layer.

Split out from the window so the file format can be checked without a display.
"""

import json
from unittest.mock import patch


class TestSettingsRoundTrip:
    def test_save_merges_rather_than_replacing(self, tmp_path):
        """Saving one section must not drop settings owned by another."""
        from mergescribe.ui import settings_store as store

        home = tmp_path / "home"
        (home / ".mergescribe").mkdir(parents=True)
        target = home / ".mergescribe" / "settings.json"
        target.write_text(json.dumps({"keep_me": 1, "trigger_key": "alt_r"}))

        with patch.object(store.Path, "home", staticmethod(lambda: home)):
            store.save_settings({"trigger_key": "cmd_r"})
            result = store.load_settings()

        assert result["keep_me"] == 1, "an untouched setting was dropped"
        assert result["trigger_key"] == "cmd_r"

    def test_load_survives_a_corrupt_file(self, tmp_path):
        from mergescribe.ui import settings_store as store

        home = tmp_path / "home"
        (home / ".mergescribe").mkdir(parents=True)
        (home / ".mergescribe" / "settings.json").write_text("{not json")

        with patch.object(store.Path, "home", staticmethod(lambda: home)):
            assert store.load_settings() == {}


class TestEnvKeys:
    def test_writing_a_key_preserves_unrelated_lines(self, tmp_path):
        """The .env may hold other people's variables; we only own our key."""
        from mergescribe.ui import settings_store as store

        home = tmp_path / "home"
        (home / ".mergescribe").mkdir(parents=True)
        env = home / ".mergescribe" / ".env"
        env.write_text("SOMETHING_ELSE=keepme\nOPENROUTER_API_KEY=old\n")

        with patch.object(store.Path, "home", staticmethod(lambda: home)):
            store.save_env_keys({"OPENROUTER_API_KEY": "new"})

        written = env.read_text()
        assert "SOMETHING_ELSE=keepme" in written
        assert "OPENROUTER_API_KEY=new" in written
        assert "old" not in written

    def test_an_empty_key_is_not_written(self, tmp_path):
        from mergescribe.ui import settings_store as store

        home = tmp_path / "home"
        (home / ".mergescribe").mkdir(parents=True)
        env = home / ".mergescribe" / ".env"
        env.write_text("OPENROUTER_API_KEY=existing\n")

        with patch.object(store.Path, "home", staticmethod(lambda: home)):
            store.save_env_keys({"OPENROUTER_API_KEY": ""})

        assert "OPENROUTER_API_KEY" not in env.read_text()


class TestModelIdParsing:
    def test_accepts_commas_newlines_and_spaces(self):
        from mergescribe.ui.settings_store import _parse_model_ids

        assert _parse_model_ids("a/b, c/d\ne/f") == ["a/b", "c/d", "e/f"]

    def test_drops_blanks_and_duplicates(self):
        from mergescribe.ui.settings_store import _parse_model_ids

        assert _parse_model_ids("a/b\n\n a/b \n c/d") == ["a/b", "c/d"]


class TestRoutingStatus:
    def test_says_so_when_there_is_no_key(self):
        from mergescribe.ui.settings_store import get_routing_status

        assert "No OpenRouter key" in get_routing_status("")

    def test_names_the_model_and_its_routing(self):
        from mergescribe.ui.settings_store import get_routing_status

        status = get_routing_status("sk-or-test", "openai/gpt-5.6-luna",
                                    ["together"], "none")
        assert "Luna" in status
        assert "together" in status
        assert "none reasoning" in status


class TestRemoveSettings:
    def test_removes_only_the_named_keys(self, tmp_path):
        """Resetting a prompt must actually delete the override, not leave it on disk."""
        from mergescribe.ui import settings_store as store

        home = tmp_path / "home"
        (home / ".mergescribe").mkdir(parents=True)
        target = home / ".mergescribe" / "settings.json"
        target.write_text(json.dumps({"system_prompt": "old override", "trigger_key": "cmd_r"}))

        with patch.object(store.Path, "home", staticmethod(lambda: home)):
            store.remove_settings(["system_prompt"])
            result = store.load_settings()

        assert "system_prompt" not in result
        assert result["trigger_key"] == "cmd_r"

    def test_missing_file_or_key_is_harmless(self, tmp_path):
        from mergescribe.ui import settings_store as store

        home = tmp_path / "home"
        with patch.object(store.Path, "home", staticmethod(lambda: home)):
            store.remove_settings(["anything"])   # no file at all

        (home / ".mergescribe").mkdir(parents=True)
        (home / ".mergescribe" / "settings.json").write_text(json.dumps({"a": 1}))
        with patch.object(store.Path, "home", staticmethod(lambda: home)):
            store.remove_settings(["not_there"])
            assert store.load_settings() == {"a": 1}


class TestCatalogue:
    def test_the_nitro_variant_is_a_known_model(self):
        """Otherwise the popup shows the default while a custom field holds the real value."""
        from mergescribe.ui.settings_store import KNOWN_CORRECTION_MODELS

        assert "openai/gpt-5.6-luna:nitro" in {slug for slug, _ in KNOWN_CORRECTION_MODELS}
