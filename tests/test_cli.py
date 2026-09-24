"""The console entry point launches the marimo app."""

from __future__ import annotations

from openqarp_demo.cli import APP, build_command, parse_args


def test_app_ships_next_to_its_stylesheet() -> None:
    assert APP.is_file()
    assert APP.with_name("theme.css").is_file()


def test_default_command_runs_the_app() -> None:
    command = build_command(parse_args([]))
    assert command[1:4] == ["-m", "marimo", "run"]
    assert command[4] == str(APP)


def test_edit_port_and_headless_flags() -> None:
    command = build_command(parse_args(["--edit", "--port", "2719", "--headless"]))
    assert command[3] == "edit"
    assert command[-3:] == ["--port", "2719", "--headless"]
