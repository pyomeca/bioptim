# Conda lockfiles used by GitHub Actions

The test workflows install explicit, platform-specific Conda lockfiles instead
of resolving `environment.yml` in every job. The CI-only test dependencies are
declared in `ci-dependencies.yml` and are resolved together with the project
environment.

The current GitHub-hosted runners require these Conda platforms:

- Ubuntu: `linux-64`
- macOS Apple Silicon: `osx-arm64`
- Windows: `win-64`

Regenerate all lockfiles from the repository root with `conda-lock 4.0.2`:

```bash
conda-lock lock \
  --kind explicit \
  --micromamba \
  --without-cuda \
  --platform linux-64 \
  --platform osx-arm64 \
  --platform win-64 \
  --file environment.yml \
  --file .github/conda-lock/ci-dependencies.yml \
  --filename-template '.github/conda-lock/conda-{platform}.lock'
```

The trusted `warm_conda_cache.yml` workflow caches the complete `bioptim`
Conda environment on `master` and refreshes it every Monday. Pull-request
workflows only restore these caches, so forked pull requests do not attempt a
cache write and all pull requests can reuse the default-branch caches.

The cache key contains the runner OS, runner architecture, lockfile hash, and
ISO week. A lockfile change therefore invalidates the cache immediately. At the
weekly refresh, the warmer restores the previous cache with the same lockfile,
reconciles it against the explicit specification, and saves a new entry. This
limits the lifetime of a potentially damaged cache without forcing a full
environment download.

If a pull request changes a lockfile, its first test run cannot use the
default-branch cache for that new hash and installs the explicit environment in
each shard. Once merged, the trusted warmer publishes the new shared cache.

When a runner image changes architecture, add a lockfile for its Conda platform
and update the corresponding workflow matrix entry.
