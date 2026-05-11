import tempfile
import subprocess
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from wheatley.config import Config, load_config
from wheatley.prompting import build_system_prompt
from wheatley.tools.builtins import build_registry
from wheatley.tools.photo import _auto_photo_command, _short_side_scale_filter
from wheatley.tools.registry import ToolCall


def _tool_cfg() -> Config:
    cfg = Config()
    profile_cfg = load_config()
    cfg.language = profile_cfg.language
    cfg.tools = profile_cfg.tools
    cfg.prompts = profile_cfg.prompts
    cfg.runtime.default_language = cfg.language.default
    cfg.tools.enabled = True
    cfg.tools.tool_settings["set_eye_expression"] = {
        **cfg.tools.tool_settings.get("set_eye_expression", {}),
        "enabled": True,
    }
    cfg.tools.tool_settings["web_search"] = {
        **cfg.tools.tool_settings.get("web_search", {}),
        "enabled": False,
    }
    cfg.tools.web_search_max_results = 5
    cfg.tools.web_search_timeout_seconds = 5.0
    return cfg


class ToolTests(unittest.TestCase):
    def test_set_eye_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tool_cfg()
            cfg.runtime.state_dir = str(Path(tmp) / "state")
            cfg.ensure_dirs()
            registry = build_registry(cfg)
            result = registry.execute(
                ToolCall("set_eye_expression", {"expression": "thinking"})
            )
            self.assertTrue(result.ok)
            self.assertTrue((Path(cfg.runtime.state_dir) / "eye.json").exists())

    def test_disabled_tool_is_not_registered(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["set_eye_expression"] = {
            **cfg.tools.tool_settings.get("set_eye_expression", {}),
            "enabled": False,
        }
        registry = build_registry(cfg)
        names = [spec.name for spec in registry.specs()]

        self.assertNotIn("set_eye_expression", names)
        result = registry.execute(ToolCall("set_eye_expression", {"expression": "happy"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.content["error"], "unknown_tool")

    def test_unknown_tool(self):
        cfg = _tool_cfg()
        registry = build_registry(cfg)
        result = registry.execute(ToolCall("missing", {}))
        self.assertFalse(result.ok)

    def test_calculator(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["calculator"] = {
            **cfg.tools.tool_settings.get("calculator", {}),
            "enabled": True,
        }
        registry = build_registry(cfg)
        result = registry.execute(
            ToolCall("calculator", {"expression": "sqrt(5+5)+sin(4)**3+6/7"})
        )
        self.assertTrue(result.ok)
        self.assertIn("result", result.content)

    def test_calculator_rejects_unsafe_expression(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["calculator"] = {
            **cfg.tools.tool_settings.get("calculator", {}),
            "enabled": True,
        }
        registry = build_registry(cfg)
        result = registry.execute(
            ToolCall("calculator", {"expression": "__import__('os').system('date')"})
        )
        self.assertFalse(result.ok)

    def test_time_tool_excludes_uptime(self):
        cfg = _tool_cfg()
        registry = build_registry(cfg)
        result = registry.execute(ToolCall("get_time", {}))

        self.assertTrue(result.ok)
        self.assertIn("iso", result.content)
        self.assertIn("timezone", result.content)
        self.assertNotIn("uptime_seconds", result.content)

    def test_system_status_includes_uptime(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["system_status"] = {
            **cfg.tools.tool_settings.get("system_status", {}),
            "enabled": True,
        }
        registry = build_registry(cfg)
        result = registry.execute(ToolCall("system_status", {}))

        self.assertTrue(result.ok)
        self.assertIn("uptime_seconds", result.content)

    def test_take_photo_is_registered_when_enabled(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["take_photo"] = {
            **cfg.tools.tool_settings.get("take_photo", {}),
            "enabled": True,
        }
        cfg.tools.photo_command = None

        registry = build_registry(cfg)

        self.assertIn("take_photo", [spec.name for spec in registry.specs()])

    def test_take_photo_uses_configured_command_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tool_cfg()
            cfg.runtime.data_dir = tmp
            cfg.tools.tool_settings["take_photo"] = {
                **cfg.tools.tool_settings.get("take_photo", {}),
                "enabled": True,
            }
            cfg.tools.photo_command = [
                "camera",
                "--short-side",
                "{short_side}",
                "--quality",
                "{quality}",
                "{output}",
            ]
            cfg.tools.photo_short_side = 320
            cfg.tools.photo_quality = 70
            registry = build_registry(cfg)

            def fake_run(command, **kwargs):
                del kwargs
                Path(command[-1]).write_bytes(b"fake-jpeg")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("wheatley.tools.photo._resize_photo_if_possible"), patch(
                "wheatley.tools.photo._photo_dimensions",
                return_value=(426, 320),
            ), patch(
                "wheatley.tools.photo.subprocess.run",
                side_effect=fake_run,
            ) as run:
                result = registry.execute(ToolCall("take_photo", {}))

            self.assertTrue(result.ok, result.content)
            self.assertEqual(result.content["width"], 426)
            self.assertEqual(result.content["height"], 320)
            self.assertEqual(result.content["short_side"], 320)
            self.assertEqual(result.content["quality"], 70)
            command = run.call_args_list[0][0][0]
            self.assertEqual(
                command[:5],
                ["camera", "--short-side", "320", "--quality", "70"],
            )
            photo_path = Path(command[-1])
            self.assertEqual(photo_path.suffix, ".jpg")
            self.assertTrue(photo_path.name.endswith("_photo.jpg"))
            relative = photo_path.relative_to(Path(tmp))
            self.assertEqual(relative.parts[0], "photos")
            self.assertRegex(str(relative), r"^photos/\d{4}/\d{2}/\d{2}/")

    def test_macos_ffmpeg_camera_fallback_uses_supported_avfoundation_mode(self):
        device_list = """
[AVFoundation indev @ 0x123] AVFoundation video devices:
[AVFoundation indev @ 0x123] [0] FaceTime HD Camera
[AVFoundation indev @ 0x123] [1] Continuity Camera
[AVFoundation indev @ 0x123] [2] Continuity Desk View Camera
[AVFoundation indev @ 0x123] AVFoundation audio devices:
"""

        def fake_run(command, **kwargs):
            del kwargs
            self.assertIn("-list_devices", command)
            return subprocess.CompletedProcess(command, 1, "", device_list)

        with patch("wheatley.tools.photo.platform.system", return_value="Darwin"), patch(
            "wheatley.tools.photo.shutil.which",
            side_effect=lambda name: "/opt/homebrew/bin/ffmpeg"
            if name == "ffmpeg"
            else None,
        ), patch(
            "wheatley.tools.photo.subprocess.run",
            side_effect=fake_run,
        ):
            command = _auto_photo_command(Path("/tmp/photo.jpg"), 640, 75)

        self.assertEqual(
            command,
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "avfoundation",
                "-pixel_format",
                "uyvy422",
                "-framerate",
                "30",
                "-video_size",
                "1080x1920",
                "-i",
                "0:none",
                "-ss",
                "2",
                "-frames:v",
                "1",
                "-vf",
                _short_side_scale_filter(640),
                "/tmp/photo.jpg",
            ],
        )

    def test_tool_description_override(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["calculator"] = {
            "enabled": True,
            "description": "Custom math tool.",
            "instructions": "Use exact math for everything.",
        }
        registry = build_registry(cfg)
        calculator = [spec for spec in registry.specs() if spec.name == "calculator"][0]
        self.assertIn("Custom math tool", calculator.description)
        self.assertIn("Use exact math", calculator.description)

    def test_web_search_brave_uses_api_provider(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["web_search"]["enabled"] = True
        body = (
            b'{"web":{"results":[{"title":"Example","url":"https://example.com",'
            b'"description":"Example result","extra_snippets":["More context"]}]}}'
        )
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test-key"}), patch(
            "wheatley.tools.web.urllib.request.urlopen",
            return_value=_FakeResponse(body, "application/json"),
        ) as urlopen:
            registry = build_registry(cfg)
            result = registry.execute(ToolCall("web_search", {"query": "example"}))
        self.assertTrue(result.ok)
        self.assertEqual(result.content["provider"], "brave")
        self.assertEqual(result.content["results"][0]["url"], "https://example.com")
        self.assertEqual(urlopen.call_count, 2)
        headers = {
            key.lower(): value
            for key, value in urlopen.call_args.args[0].headers.items()
        }
        self.assertEqual(headers["x-subscription-token"], "test-key")

    def test_web_search_is_not_registered_without_internet(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["web_search"]["enabled"] = True
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test-key"}), patch(
            "wheatley.tools.web.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            registry = build_registry(cfg)

        names = [spec.name for spec in registry.specs()]
        self.assertNotIn("web_search", names)
        self.assertNotIn("web_search", build_system_prompt(cfg, registry))
        result = registry.execute(ToolCall("web_search", {"query": "example"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.content["error"], "unknown_tool")

    def test_web_search_is_not_registered_without_api_key(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["web_search"]["enabled"] = True
        with patch.dict("os.environ", {}, clear=True), patch(
            "wheatley.tools.web.urllib.request.urlopen",
        ) as urlopen:
            registry = build_registry(cfg)

        names = [spec.name for spec in registry.specs()]
        self.assertNotIn("web_search", names)
        urlopen.assert_not_called()

    def test_python_interpreter_uses_preamble_input_and_read_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp)
            (profile / "files").mkdir()
            (profile / "files" / "data.json").write_text(
                '{"items": ["a", "b", "a"]}', encoding="utf-8"
            )
            (profile / "python_preamble.py").write_text(
                "from collections import Counter\n"
                "def top(items):\n"
                "    return Counter(items).most_common()\n",
                encoding="utf-8",
            )
            cfg = _tool_cfg()
            cfg.profile_dir = str(profile)
            cfg.tools.tool_settings["python_interpreter"] = {
                **cfg.tools.tool_settings.get("python_interpreter", {}),
                "enabled": True,
            }
            cfg.tools.python_interpreter_read_roots = ["files"]
            registry = build_registry(cfg)

            result = registry.execute(
                ToolCall(
                    "python_interpreter",
                    {
                        "input": {"extra": "c"},
                        "code": (
                            "data = read_json('data.json')\n"
                            "result = {'top': top(data['items']), "
                            "'files': list_files('*.json'), "
                            "'extra': input['extra']}"
                        ),
                    },
                )
            )

            self.assertTrue(result.ok, result.content)
            self.assertEqual(result.content["result"]["top"][0], ["a", 2])
            self.assertEqual(result.content["result"]["files"], ["data.json"])
            self.assertEqual(result.content["result"]["extra"], "c")

    def test_python_interpreter_rejects_model_imports(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["python_interpreter"] = {
            **cfg.tools.tool_settings.get("python_interpreter", {}),
            "enabled": True,
        }
        registry = build_registry(cfg)

        result = registry.execute(
            ToolCall("python_interpreter", {"code": "import os\nresult = 1"})
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.content["error"], "blocked_node: Import")
        self.assertEqual(result.content["phase"], "validation")

    def test_python_interpreter_blocks_read_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp)
            (profile / "files").mkdir()
            (profile / "secret.txt").write_text("secret", encoding="utf-8")
            cfg = _tool_cfg()
            cfg.profile_dir = str(profile)
            cfg.tools.tool_settings["python_interpreter"] = {
                **cfg.tools.tool_settings.get("python_interpreter", {}),
                "enabled": True,
            }
            cfg.tools.python_interpreter_read_roots = ["files"]
            registry = build_registry(cfg)

            result = registry.execute(
                ToolCall(
                    "python_interpreter",
                    {"code": "result = read_text('../secret.txt')"},
                )
            )

            self.assertFalse(result.ok)
            self.assertIn("parent directory paths", result.content["error"])
            self.assertEqual(result.content["phase"], "code")

    def test_python_interpreter_reports_missing_result(self):
        cfg = _tool_cfg()
        cfg.tools.tool_settings["python_interpreter"] = {
            **cfg.tools.tool_settings.get("python_interpreter", {}),
            "enabled": True,
        }
        registry = build_registry(cfg)

        result = registry.execute(
            ToolCall("python_interpreter", {"code": "value = 42"})
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.content["error"], "missing_result")
        self.assertEqual(result.content["phase"], "code")

    def test_python_interpreter_reports_preamble_phase_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp)
            (profile / "files").mkdir()
            (profile / "python_preamble.py").write_text(
                "raise RuntimeError('bad preamble')\n",
                encoding="utf-8",
            )
            cfg = _tool_cfg()
            cfg.profile_dir = str(profile)
            cfg.tools.tool_settings["python_interpreter"] = {
                **cfg.tools.tool_settings.get("python_interpreter", {}),
                "enabled": True,
            }
            registry = build_registry(cfg)

            result = registry.execute(
                ToolCall("python_interpreter", {"code": "result = 1"})
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.content["phase"], "preamble")
            self.assertIn("bad preamble", result.content["error"])


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        content_type: str,
        url: str = "https://example.com",
    ) -> None:
        self.body = body
        self.headers = {"Content-Type": content_type}
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]

    def geturl(self) -> str:
        return self.url


if __name__ == "__main__":
    unittest.main()
