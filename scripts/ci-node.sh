#!/usr/bin/env bash
# Start a throwaway Gnarl node for the integration tests and export GNARL_URL.
#
# Downloads the latest public release (gnarl-dev/releases, the binary a user
# installs), verifies it against the release's SHA256SUMS, and starts it
# single-node, plain HTTP, on a temp data dir. Memory needs the node's
# on-device embedding model; the node fetches it on first use.
#
# Usage (GitHub Actions): scripts/ci-node.sh   — needs GH_TOKEN for `gh`.
set -euo pipefail

port="${GNARL_TEST_PORT:-43900}"
work="${RUNNER_TEMP:-$(mktemp -d)}/gnarl-node"
mkdir -p "$work" && cd "$work"

case "$(uname -m)" in
  x86_64)        triple=x86_64-unknown-linux-gnu ;;
  aarch64|arm64) triple=aarch64-unknown-linux-gnu ;;
  *) echo "::error::no released node for $(uname -m)"; exit 1 ;;
esac
asset="gnarl-${triple}.tar.gz"
tag=$(gh release view --repo gnarl-dev/releases --json tagName -q .tagName)
echo "node: gnarl-dev/releases $tag ($asset)"
gh release download "$tag" --repo gnarl-dev/releases --pattern "$asset" --pattern SHA256SUMS.txt --clobber
grep " \./${asset}\$\| ${asset}\$" SHA256SUMS.txt | sed 's# \./# #' | sha256sum -c -
tar xzf "$asset"
test -x gnarl

./gnarl start --data-dir "$work/data" --port "$port" --udp-port $((port + 1)) \
  --single-node --no-tls --headless > "$work/node.log" 2>&1 &
echo $! > "$work/pid"

url="http://127.0.0.1:${port}"
for _ in $(seq 1 60); do
  if curl -fsS "$url/v1/node/status" > /dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "$url/v1/node/status" > /dev/null || { tail -50 "$work/node.log"; exit 1; }

# One memory round trip, so a node that cannot embed fails here, by name,
# rather than as a confusing assertion in the first test.
curl -fsS -X POST "$url/v1/memory/remember" -H 'content-type: application/json' \
  -d '{"content":"ci warm-up","namespace":"ci-warmup"}' | tee /dev/stderr | grep -q 'semantic=true' \
  || { echo "::error::node is not embedding with its semantic model"; tail -50 "$work/node.log"; exit 1; }

echo "GNARL_URL=$url" >> "${GITHUB_ENV:-/dev/null}"
echo "GNARL_URL=$url"
