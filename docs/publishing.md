# Publishing pros-sketch

The distribution is `pros-sketch`, while the Python import is `pros`.
The workflow `.github/workflows/publish.yml` publishes to PyPI only when a
GitHub Release is published. Manual workflow runs perform validation and build
artifacts without uploading. No PyPI API token secret is required.

## One-time PyPI setup

For the first release, sign in to https://pypi.org/manage/account/publishing/
and add a pending publisher under GitHub:

| Field | Value |
| --- | --- |
| PyPI Project Name | `pros-sketch` |
| Owner | `LeiLi-Uchicago` |
| Repository name | `PROS` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Use the workflow filename, not its display title or full path. A pending
publisher does not reserve the project name; successful first upload creates
it. If this project already belongs to your account, add the publisher from
the project's Publishing settings instead.

In GitHub repository Settings → Environments, create the `pypi` environment.
Its name must match both the workflow and PyPI configuration. Optional review
and deployment restrictions can be configured there.

## Release procedure

1. Commit and push the reviewed code and publishing workflow to GitHub.
2. Run **Publish to PyPI** manually from Actions. This tests Python 3.10–3.12,
   builds the sdist and wheel, checks metadata, and tests the installed wheel
   outside the checkout. Confirm the run succeeds.
3. Ensure `pyproject.toml` and `pros/__init__.py` specify the same release
   version. For the first release this is `0.1.0`.
4. Create a GitHub Release with tag `v0.1.0` pointing to the validated commit.
   Publishing it triggers the workflow, repeats validation, and uploads the
   built distributions through Trusted Publishing. A draft release does not.
5. Verify the project page and install `pros-sketch==0.1.0` in a new environment.
   For AnnData support, install `pros-sketch[adata]==0.1.0`.

For later releases, update both version values and use the corresponding tag.
PyPI does not allow overwriting previously uploaded distribution filenames.
If an upload fails, inspect the logs and which artifacts reached PyPI before
retrying; this workflow intentionally does not hide existing-file errors.

TestPyPI is not configured by this workflow. It requires a separate account
and trusted publisher if a rehearsal upload is wanted.

## Scope of validation

The workflow tests core and AnnData functionality. The scSampler adapter is
covered by contract tests, not an end-to-end installation of scSampler. Large
benchmark datasets and manuscript runtime reproduction are not run in CI.

References:
- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- https://github.com/pypa/gh-action-pypi-publish
