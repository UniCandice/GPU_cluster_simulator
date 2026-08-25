"""The CLI ends where the reviewer looks: at a rendered page.

`gcsim run` used to stop at a Parquet path -- simulate, print the diagnosis,
write the run, done. Every other entry point (run_all.py, `gcsim dashboard`)
finishes by building and opening the dashboard, and `run` is the command the
README teaches for single scenarios, so it was exactly the one a fresh clone
would try and find wanting.
"""

from __future__ import annotations

from gcsim.cli import main


def test_run_finishes_with_a_dashboard(bundle, tmp_path):
    rc = main(["run", "--scenario", "healthy", "--mesh", "coarse", "--seed", "7",
               "--out", str(tmp_path / "runs"), "--no-open"])
    assert rc == 0

    #  A custom runs directory gets a sibling dashboard, never the repo's own.
    #  Every seed builds its own standalone page; no index.html is written.
    page = tmp_path / "dashboard" / "index_seed7.html"
    assert page.exists()
    assert not (tmp_path / "dashboard" / "index.html").exists()
    html = page.read_text(encoding="utf-8")
    assert "healthy__coarse" in html
    assert '"seed":7' in html


def test_no_dashboard_restores_the_write_only_behaviour(bundle, tmp_path):
    rc = main(["run", "--scenario", "healthy", "--mesh", "coarse", "--seed", "7",
               "--out", str(tmp_path / "runs"), "--no-dashboard"])
    assert rc == 0
    assert (tmp_path / "runs").exists()
    assert not (tmp_path / "dashboard").exists()


def test_no_write_builds_nothing_and_does_not_raise(bundle, tmp_path):
    rc = main(["run", "--scenario", "healthy", "--mesh", "coarse", "--seed", "7",
               "--no-write", "--no-open"])
    assert rc == 0
    assert not (tmp_path / "dashboard").exists()
