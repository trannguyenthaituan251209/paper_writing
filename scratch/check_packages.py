import sys
print("Python executable:", sys.executable)

for pkg in ['sklearn', 'scipy', 'torch', 'transformers', 'pyvi', 'matplotlib', 'seaborn']:
    try:
        mod = __import__(pkg)
        print(f"  [+] {pkg}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"  [-] {pkg}: NOT installed")
