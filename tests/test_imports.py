"""Keep saved-latent analysis independent of inference and plotting imports."""
import subprocess
import sys


def test_analysis_imports_without_inference_or_plotting(tmp_path):
    code = """
import sys
import graphcast_latent
import graphcast_latent.graph
import graphcast_latent.spectral
import graphcast_latent.storage
for name in ('jax', 'jaxlib', 'haiku', 'graphcast', 'hisgraphcast', 'matplotlib', 'cartopy'):
    assert not any(key == name or key.startswith(name + '.') for key in sys.modules), name
"""
    subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, check=True)
