"""Config schema, loader, and environment overrides."""

import tempfile
import unittest
from pathlib import Path

from configs.loader import CONFIG_PATH, ConfigError, load_config, validate_config


class TestConfig(unittest.TestCase):
    def test_repo_config_is_valid(self):
        cfg = load_config(CONFIG_PATH, environ={})
        self.assertEqual(cfg["agent"]["mode"], "agent")
        self.assertEqual(cfg["serving"]["host"], "127.0.0.1")

    def test_defaults_fill_missing_sections(self):
        cfg = validate_config({})
        self.assertEqual(cfg["agent"]["top_k"], 10)
        self.assertEqual(cfg["llm"]["num_ctx"], 4096)
        self.assertTrue(cfg["serving"]["auth_enabled"])

    def test_unknown_key_rejected(self):
        with self.assertRaises(ConfigError) as cm:
            validate_config({"agent": {"rerank_candidate": 10}})
        self.assertIn("rerank_candidate", str(cm.exception))

    def test_out_of_range_rejected(self):
        for raw in (
            {"agent": {"top_k": 0}},
            {"agent": {"mode": "turbo"}},
            {"llm": {"provider": "openai"}},
            {"elasticsearch": {"port": 70000}},
        ):
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                validate_config(raw)

    def test_cross_field_rules(self):
        with self.assertRaises(ConfigError):
            validate_config({"agent": {"top_k": 20, "per_method_top_k": 10}})
        with self.assertRaises(ConfigError):
            validate_config({"agent": {"request_timeout_s": 300}, "serving": {"hard_timeout_s": 60}})

    def test_env_overrides(self):
        cfg = load_config(
            CONFIG_PATH,
            environ={"ES_HOST": "es.internal", "ES_PORT": "9300", "LLM_MODEL": "other:7b"},
        )
        self.assertEqual(cfg["elasticsearch"]["host"], "es.internal")
        self.assertEqual(cfg["elasticsearch"]["port"], 9300)
        self.assertEqual(cfg["llm"]["model"], "other:7b")

    def test_config_path_from_env_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("agent:\n  top_k: 5\n")
            cfg = load_config(environ={"AGENTIC_RAG_CONFIG": str(path)})
            self.assertEqual(cfg["agent"]["top_k"], 5)

            path.write_text("- not\n- a mapping\n")
            with self.assertRaises(ConfigError):
                load_config(path, environ={})
            path.write_text("agent: [unclosed\n")
            with self.assertRaises(ConfigError):
                load_config(path, environ={})
        with self.assertRaises(ConfigError):
            load_config(Path(tmp) / "missing.yaml", environ={})


if __name__ == "__main__":
    unittest.main()
