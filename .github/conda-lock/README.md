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

Then update the source fingerprint and run the consistency check:

```bash
python .github/conda-lock/check_lockfiles.py --print-source-hash \
  > .github/conda-lock/source-files.sha256
python .github/conda-lock/check_lockfiles.py
```

The check runs before Conda setup in every test and cache-warming job. It fails
when either source file changes without updating the fingerprint, and also
validates each lockfile's platform, input hash, explicit marker, package URLs,
and package checksums. The fingerprint is an accidental-staleness guard rather
than a security boundary: changes to it must be reviewed together with the
generated lockfile changes.

Trusted cache producers run only on `master`. The `warm_conda_cache.yml`
workflow prepares macOS and Windows after an environment-related push, and
refreshes all three platforms every Monday. Pull-request workflows only restore
these caches, so forked pull requests do not attempt a cache write and all pull
requests can reuse the default-branch caches.

The Linux test workflow also runs on pushes to `master`, preserving the default
branch status used by the README badge. Its test shards wait for a trusted
Linux cache-preparation job, so an environment change is installed and cached
once before the six shards start. On the same environment-related push, the
separate warmer handles only macOS and Windows. Weekly and manual warming still
cover all three platforms. The macOS/Windows test shards remain pull-request
only to avoid twelve additional post-merge jobs.

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
