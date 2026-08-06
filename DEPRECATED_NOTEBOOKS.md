The following notebooks are deprecated. Use notebooks/colab_first_creative_video_supported.ipynb as the single supported Colab entrypoint.

Deprecated:
- notebooks/colab_first_creative_video.ipynb
- notebooks/colab_first_creative_video_full.ipynb

Rationale:
All environment setup, CUDA detection, venv creation, dependency installation, and orchestration have been moved into scripts/colab_run_full_pipeline.py. The supported notebook delegates to that script and only displays progress and the final output.