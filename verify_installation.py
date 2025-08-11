#!/usr/bin/env python3

import importlib
import sys

def check_dependency(name, package_name=None):
    package_name = package_name or name
    try:
        importlib.import_module(package_name)
        print(f"✅ {name}")
        return True
    except ImportError:
        print(f"❌ {name} - not installed")
        return False

def check_version(package_name, min_version=None):
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
            print(f"   Version: {version}")
            if min_version:
                from packaging import version as pkg_version
                if pkg_version.parse(version) >= pkg_version.parse(min_version):
                    print(f"   ✅ Version {version} >= {min_version}")
                else:
                    print(f"   ⚠️  Version {version} < {min_version} (recommended)")
        return True
    except Exception as e:
        print(f"   Error checking version: {e}")
        return False

def main():
    print("🔍 Checking MergeScribe Dependencies...")
    print("=" * 50)
    
    print("\n📦 Core Dependencies:")
    check_dependency("numpy")
    check_dependency("scipy")
    check_dependency("soundfile")
    check_dependency("sounddevice")
    
    print("\n🤖 AI/ML Dependencies:")
    check_dependency("groq")
    check_version("groq", "0.31.0")
    check_dependency("openai")
    check_dependency("google_genai")
    
    print("\n🖥️ UI Dependencies:")
    check_dependency("rumps")
    check_dependency("flet")
    check_dependency("pynput")
    check_dependency("pyautogui")
    
    print("\n🍎 macOS Dependencies:")
    check_dependency("Quartz", "Quartz")
    
    print("\n🏠 Local AI:")
    check_dependency("parakeet_mlx")
    check_dependency("mlx.core", "mlx")
    
    print("\n🛠️ Development Tools:")
    check_dependency("pytest")
    check_dependency("ruff")
    
    print("\n" + "=" * 50)
    print("✅ Dependency check complete!")
    
    print("\n🧪 Testing Providers...")
    try:
        from providers import get_providers
        providers = get_providers()
        print(f"✅ Found {len(providers)} providers")
        for provider in providers:
            print(f"   - {provider.__name__}")
    except Exception as e:
        print(f"❌ Error loading providers: {e}")

if __name__ == "__main__":
    main()
