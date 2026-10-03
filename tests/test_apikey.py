"""The key resolver must fail loudly; a silent fallback would ship a stale or absent key."""

import unittest
from unittest.mock import patch

from voxscribe.apikey import ENV_VAR, ApiKeyError, resolve_api_key


class ResolveApiKeyTest(unittest.TestCase):
    def test_command_wins_over_environment(self) -> None:
        config = {"api_key_command": "sh -c 'printf from-command'"}
        with patch.dict("os.environ", {ENV_VAR: "from-environment"}):
            self.assertEqual(resolve_api_key(config), "from-command")

    def test_failing_command_never_falls_back_to_environment(self) -> None:
        config = {"api_key_command": "false"}
        with patch.dict("os.environ", {ENV_VAR: "from-environment"}):
            with self.assertRaises(ApiKeyError):
                resolve_api_key(config)

    def test_command_printing_nothing_is_an_error(self) -> None:
        with self.assertRaises(ApiKeyError):
            resolve_api_key({"api_key_command": "true"})

    def test_missing_command_falls_back_to_environment(self) -> None:
        with patch.dict("os.environ", {ENV_VAR: "from-environment"}):
            self.assertEqual(resolve_api_key({}), "from-environment")

    def test_no_command_and_no_environment_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ApiKeyError):
                resolve_api_key({"api_key_command": ""})


if __name__ == "__main__":
    unittest.main()
