#!/bin/bash
# Stage and submit the phase-matched L3870N4096 resolution campaign.

set -euo pipefail

LOW="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000"
HIGH="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N4096/000"
REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
CAMPAIGN="${REPO}/scripts/leonardo/l3870_n4096_resolution"

if [[ ! -d "${LOW}" || ! -x "${LOW}/pinocchio.x" ]]; then
	echo "Low-resolution source run is unavailable: ${LOW}" >&2
	exit 2
fi
if [[ ! -f "${CAMPAIGN}/submit_pinocchio.sh" ]]; then
	echo "Campaign scripts are unavailable: ${CAMPAIGN}" >&2
	exit 2
fi
mkdir -p "${HIGH}/logs"
if [[ -e "${HIGH}/pipeline_jobs.env" ]]; then
	echo "A campaign was already submitted; inspect ${HIGH}/pipeline_jobs.env" >&2
	exit 2
fi
if compgen -G "${HIGH}/pinocchio.*" >/dev/null; then
	echo "The high-resolution directory already contains PINOCCHIO outputs" >&2
	exit 2
fi

cp "${LOW}/outputs" "${HIGH}/outputs"
cp "${LOW}/pinocchio.x" "${HIGH}/pinocchio.x"
ln -sfn "${LOW}/CambFiles" "${HIGH}/CambFiles"
sed \
	-e 's/^GridSize[[:space:]].*/GridSize               4096         % number of grid points per side/' \
	-e 's/^MaxMem[[:space:]].*/MaxMem                13000/' \
	-e 's/^% DoNotWriteHistories/DoNotWriteHistories/' \
	"${LOW}/params.txt" >"${HIGH}/params.txt"
cat >>"${HIGH}/params.txt" <<'EOF'

# Preserve the L3870N2160 observer and cone orientation.
PLCProvideConeData
PLCCenter 415.2600210416666 3786.244397541667 2252.2445095416665
PLCAxis -0.822556930095597 -0.112935946744643 -0.55735587256671
EOF

{
	echo "staged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "source_run=${LOW}"
	echo "source_binary_sha256=$(sha256sum "${LOW}/pinocchio.x" | awk '{print $1}')"
	echo "high_binary_sha256=$(sha256sum "${HIGH}/pinocchio.x" | awk '{print $1}')"
	echo "geppetto_commit=$(git -C "${REPO}" rev-parse HEAD)"
	echo "geppetto_diff_sha256=$(git -C "${REPO}" diff --no-ext-diff | sha256sum | awk '{print $1}')"
	echo "comparison_script_sha256=$(sha256sum "${REPO}/examples/compare_angular_power_resolutions.py" | awk '{print $1}')"
	echo "campaign_scripts_sha256=$(find "${CAMPAIGN}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 cat | sha256sum | awk '{print $1}')"
	echo "grid_size=4096"
	echo "mpi_ranks=4096"
	echo "nodes=128"
} >"${HIGH}/campaign_provenance.env"

cd "${HIGH}"
pinocchio_job=$(sbatch --parsable "${CAMPAIGN}/submit_pinocchio.sh")
geppetto_job=$(sbatch --parsable --dependency="afterok:${pinocchio_job}" "${CAMPAIGN}/submit_geppetto.sh")
theory_job=$(sbatch --parsable --dependency="afterok:${geppetto_job}" "${CAMPAIGN}/submit_theory.sh")
compare_job=$(sbatch --parsable --dependency="afterok:${theory_job}" "${CAMPAIGN}/submit_compare.sh")

{
	echo "PINOCCHIO_JOB_ID=${pinocchio_job}"
	echo "GEPPETTO_JOB_ID=${geppetto_job}"
	echo "THEORY_JOB_ID=${theory_job}"
	echo "COMPARE_JOB_ID=${compare_job}"
} >pipeline_jobs.env
cat pipeline_jobs.env
