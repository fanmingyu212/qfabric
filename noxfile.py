import nox

nox.options.default_venv_backend = "uv"


@nox.session
def tests(session):
    """Run the test suite."""
    if "skip-install" not in session.posargs:
        session.install("-e", ".[dev]")
    session.run("pytest")


@nox.session
def lint(session):
    """Run isort and black."""
    if "skip-install" not in session.posargs:
        session.install("-e", ".[dev]")
    session.run("black", "src/qfabric")
    session.run("isort", "src/qfabric")


@nox.session
def docs(session):
    """Build the Sphinx HTML documentation."""
    if "skip-install" not in session.posargs:
        session.install("-e", ".[dev]")
    session.run(
        "sphinx-build",
        "-b",
        "html",
        "docs/source",
        "docs/build/html",
    )


@nox.session
def docs_live(session):
    """Live-reload docs while editing."""
    if "skip-install" not in session.posargs:
        session.install("-e", ".[dev]")
    session.run(
        "sphinx-autobuild",
        "-E",
        "docs/source",
        "docs/build/html",
        "--watch",
        "src/qfabric",
        "--port",
        "0",
    )
