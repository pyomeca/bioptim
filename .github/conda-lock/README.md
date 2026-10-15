# Conda lockfiles used by GitHub Actions

The test workflows install explicit, platform-specific Conda lockfiles instead
of resolving `environment.yml` in every job. The CI-only test dependencies are
declared in `ci-dependencies.yml` and are resolved together with the project
environment.

The current GitHub-hosted runners require these Conda platforms:

- Ubuntu: `linux-64`
- macOS Apple Silicon: `osx-arm64`
- Windows: `win-64`

The Linux lock uses the Haswell (`x86_64_v3`) build of BLASFEO 0.1.4.3.
Its Intel Core (`x86_64`) build makes FATROP 0.0.4 fail the pendulum solve
before convergence. The CI dependency pin and `virtual-packages.yml` select
this variant and its matching `_x86_64-microarch-level` dependency during
regeneration. The Linux runner must support AVX2; the other platform locks
keep their own variants.

## Automatic regeneration on pull requests

`regenerate_conda_lockfiles.yml` checks every opened, updated or reopened PR,
including drafts. A single Ubuntu job checks the committed source fingerprint
and all three lockfiles. When they are current, it skips Python setup,
conda-lock installation and dependency resolution. Otherwise it resolves all
three platforms once, updates the fingerprint and validates the generated files.
The generator uses `conda-lock 4.0.2` and Micromamba `2.5.0-2` explicitly;
an unpinned Micromamba auto-download crashed with SIGSEGV in the Linux validation.
It does not run a solver in each test shard. New pushes cancel an older run for
the same PR. The workflow can also be launched manually with `workflow_dispatch`.

The job checks out the PR head and uses read-only repository permissions, so it
also works for PRs from forks.
[GitHub limits fork PR tokens to read-only access][fork-permissions].
The workflow publishes a `conda-lockfiles` artifact containing
the three generated lockfiles, their source fingerprint, the source commit and a
Git patch. No extra secret or write token is needed.

A stale-lockfile run deliberately fails **after** publishing the valid update:
the generated package versions must be reviewed and committed to make the PR
reproducible. Download the artifact from the run summary and apply its patch
from the repository root:

```bash
gh run download RUN_ID --repo OWNER/REPOSITORY --name conda-lockfiles --dir lockfile-update
git apply --check lockfile-update/conda-lockfiles.patch
git apply lockfile-update/conda-lockfiles.patch
python .github/conda-lock/check_lockfiles.py
git add .github/conda-lock/conda-*.lock .github/conda-lock/source-files.sha256
git commit -m "ci: refresh Conda lockfiles"
git push
```

Use the repository that owns the run in `--repo`. The source commit is recorded
in the artifact; regenerate against the current branch head if it has advanced.
Once the patch is committed, the next run takes the fast path and performs no
resolution. Existing test jobs continue to require the committed lockfiles to
be consistent before installing their environments.

## Manual regeneration

Regenerate all lockfiles from the repository root with `conda-lock 4.0.2`:

```bash
conda-lock lock \
  --kind explicit \
  --micromamba \
  --virtual-package-spec .github/conda-lock/virtual-packages.yml \
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

The source fingerprint normalizes CRLF to LF, so Git checkouts on Windows
produce the same hash as Linux and macOS. The check runs before Conda setup
in every test and cache-warming job. It fails when a source file changes
without updating the fingerprint, and also
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

[fork-permissions]: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflows-in-forked-repositories
