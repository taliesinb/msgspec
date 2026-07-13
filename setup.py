import os
import platform
import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.extension import Extension

# The `msgspec.javascript` bundle (the `@msgspec/msgpack` JS package) lives at
# the repo top level. Copy it into the built package as `javascript_bundle/` so
# it ships in the wheel and `msgspec.javascript.bundle_path()` can find it.
_JS_BUNDLE_SRC = Path(__file__).parent / "javascript"
_JS_BUNDLE_ITEMS = ("src", "package.json", "README.md")


class build_py(_build_py):
    def run(self):
        super().run()
        if not (_JS_BUNDLE_SRC / "src" / "msgpack.mjs").is_file():
            return
        dest = Path(self.build_lib) / "msgspec" / "javascript_bundle"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for item in _JS_BUNDLE_ITEMS:
            src = _JS_BUNDLE_SRC / item
            if src.is_dir():
                shutil.copytree(src, dest / item)
            elif src.is_file():
                shutil.copy2(src, dest / item)

# Check for 32-bit windows builds, which currently aren't supported. We can't
# rely on `platform.architecture` here since users can still run 32-bit python
# builds on 64 bit architectures.
if sys.platform == "win32" and sys.maxsize == (2**31 - 1):
    import textwrap

    error = """
    ====================================================================
    `msgspec` currently doesn't support 32-bit Python windows builds. If
    this is important for your use case, please comment on this issue:

    https://github.com/msgspec/msgspec/issues/845
    ====================================================================
    """
    print(textwrap.dedent(error))
    exit(1)


SANITIZE = os.environ.get("MSGSPEC_SANITIZE", False)
COVERAGE = os.environ.get("MSGSPEC_COVERAGE", False)
DEBUG = os.environ.get("MSGSPEC_DEBUG", SANITIZE or COVERAGE)

extra_compile_args = []
extra_link_args = []
if SANITIZE:
    extra_compile_args.extend(["-fsanitize=address", "-fsanitize=undefined"])
    extra_link_args.extend(["-lasan", "-lubsan"])
if COVERAGE:
    extra_compile_args.append("--coverage")
    extra_link_args.append("-lgcov")
if DEBUG:
    extra_compile_args.extend(["-O0", "-g", "-UNDEBUG"])
elif sys.platform != "win32":
    extra_compile_args.extend(["-g0"])
    if sys.platform == "darwin" and platform.machine().lower() == "arm64":
        extra_compile_args.extend(["-flto=thin"])
        extra_link_args.extend(["-flto=thin"])

# from https://py-free-threading.github.io/faq/#im-trying-to-build-a-library-on-windows-but-msvc-says-c-atomic-support-is-not-enabled
if sys.platform == "win32":
    extra_compile_args.extend(
        [
            "/std:c11",
            "/experimental:c11atomics",
        ]
    )

libraries = []
if sys.platform != "win32":
    libraries.append("m")

ext_modules = [
    Extension(
        "msgspec._core",
        [os.path.join("src", "msgspec", "_core.c")],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_py": build_py},
)
