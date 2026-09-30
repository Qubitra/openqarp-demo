#!/usr/bin/env bash
# Publish the current commit to a Hugging Face Docker Space: the repository as it stands,
# with this directory's README (the Space's configuration) in place of the project's own.
#
#   HF_TOKEN=hf_... deploy/hf-space/push.sh <owner>/<space>
set -euo pipefail

space="${1:?usage: push.sh <owner>/<space>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with write access to the Space}"

root="$(git rev-parse --show-toplevel)"
commit="$(git -C "$root" rev-parse --short HEAD)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git -C "$root" archive HEAD | tar -x -C "$work"
cp "$root/deploy/hf-space/README.md" "$work/README.md"
cp "$root/README.md" "$work/PROJECT.md"

cd "$work"
git init -q -b main
git add -A
git -c user.name="Qubitra" -c user.email="engineering@qubitra.io" commit -q -m "Deploy ${commit}"
git push -q --force "https://user:${HF_TOKEN}@huggingface.co/spaces/${space}" main
echo "pushed ${commit} to https://huggingface.co/spaces/${space}"
