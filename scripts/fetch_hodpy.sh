#!/bin/bash
# Pin optional upstream sources and data without modifying an existing checkout.
set -euo pipefail
DEST=${1:-outputs/galaxy_lightcone/vendor/hodpy}
REV=d303bef896fe6a92593d815f640f02865f11df60
if [[ ! -d "${DEST}/.git" ]]; then
    git clone https://github.com/amjsmith/hodpy.git "${DEST}"
    git -C "${DEST}" switch --detach "${REV}"
fi
ACTUAL=$(git -C "${DEST}" rev-parse HEAD)
if [[ "${ACTUAL}" != "${REV}" ]]; then
    printf 'Expected hodpy %s; existing checkout is %s. Use a separate destination.\n' "${REV}" "${ACTUAL}" >&2
    exit 1
fi
git -C "${DEST}" diff --exit-code
printf 'hodpy %s available at %s\n' "${REV}" "${DEST}"
