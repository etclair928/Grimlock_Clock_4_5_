#!/usr/bin/env python3
"""Health check for Grimlock - uses modern APIs only"""

import sys
from importlib.metadata import version, PackageNotFoundError, distributions


def check_package(pkg_name, min_version=None):
    """Check if package is installed with modern importlib"""
    try:
        ver = version(pkg_name)
        status = "✅"
        msg = f"{pkg_name}=={ver}"

        if min_version:
            from packaging import version as ver_compare
            if ver_compare.parse(ver) < ver_compare.parse(min_version):
                status = "⚠️"
                msg += f" (need >= {min_version})"

        return status, msg
    except PackageNotFoundError:
        return "❌", f"{pkg_name} NOT FOUND"


def main():
    print("\n" + "=" * 60)
    print("GRIMLOCK HEALTH CHECK (Modern API)")
    print("=" * 60)

    # Critical packages
    critical = [
        ("tensorflow", "2.15.0"),
        ("tensorflow-hub", None),
        ("basic-pitch", "0.4.0"),
        ("librosa", None),
        ("numpy", None),
        ("setuptools", None),
    ]

    print("\n📦 CRITICAL PACKAGES:")
    for pkg, min_ver in critical:
        status, msg = check_package(pkg, min_ver)
        print(f"  {status} {msg}")

    # Optional packages
    optional = [
        ("madmom", None),
        ("crepe", None),
        ("demucs", None),
        ("torch", None),
        ("scikit-learn", None),
    ]

    print("\n📦 OPTIONAL PACKAGES:")
    for pkg, min_ver in optional:
        status, msg = check_package(pkg, min_ver)
        print(f"  {status} {msg}")

    # Check for pkg_resources (should NOT be used)
    print("\n🔍 LEGACY API CHECK:")
    try:
        import pkg_resources
        print("  ⚠️ pkg_resources FOUND (deprecated - should be removed)")
    except ImportError:
        print("  ✅ pkg_resources not found (good - using modern APIs)")

    # Test actual imports
    print("\n🔧 IMPORT TESTS:")

    tests = [
        ("tensorflow", "tf"),
        ("tensorflow_hub", "hub"),
        ("basic_pitch", "bp"),
        ("librosa", "librosa"),
    ]

    for module_name, short_name in tests:
        try:
            module = __import__(module_name)
            if hasattr(module, "__version__"):
                ver = module.__version__
            else:
                ver = "unknown"
            print(f"  ✅ {short_name}: {ver}")
        except ImportError as e:
            print(f"  ❌ {short_name}: {e}")

    print("\n" + "=" * 60)
    print("✅ Health check complete")
    print("=" * 60)


if __name__ == "__main__":
    main()