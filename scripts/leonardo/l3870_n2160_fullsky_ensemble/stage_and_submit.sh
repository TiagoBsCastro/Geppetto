#!/bin/bash
# Stage and submit eight independent full-sky L3870N2160 realizations.

set -euo pipefail

SOURCE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000"
BASE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/fullsky_z0493_ensemble"
REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
CAMPAIGN="${REPO}/scripts/leonardo/l3870_n2160_fullsky_ensemble"
SEEDS_FILE="${CAMPAIGN}/seeds.txt"

mapfile -t SEEDS <"${SEEDS_FILE}"
if ((${#SEEDS[@]} != 8)); then
	echo "Expected exactly eight seeds in ${SEEDS_FILE}" >&2
	exit 2
fi
if [[ ! -x "${SOURCE}/pinocchio.x" || ! -f "${SOURCE}/params.txt" ]]; then
	echo "Source PINOCCHIO run is incomplete: ${SOURCE}" >&2
	exit 2
fi
if [[ -e "${BASE}/pipeline_jobs.env" ]]; then
	echo "Campaign was already submitted: ${BASE}/pipeline_jobs.env" >&2
	exit 2
fi
mkdir -p "${BASE}/logs"

for seed in "${SEEDS[@]}"; do
	printf -v label 'seed%06d' "${seed}"
	run="${BASE}/${label}"
	if compgen -G "${run}/pinocchio.000.massmap.seg*.fits" >/dev/null; then
		echo "Refusing to overwrite existing outputs in ${run}" >&2
		exit 2
	fi
	mkdir -p "${run}/logs"
	ln -sfn "${SOURCE}/pinocchio.x" "${run}/pinocchio.x"
	ln -sfn "${SOURCE}/CambFiles" "${run}/CambFiles"
	sed \
		-e "s/^RandomSeed[[:space:]].*/RandomSeed             ${seed}/" \
		-e 's/^StartingzForPLC[[:space:]].*/StartingzForPLC        0.492562563187/' \
		-e 's/^PLCAperture[[:space:]].*/PLCAperture            180/' \
		-e 's/^MaxMem[[:space:]].*/MaxMem                14000/' \
		-e 's/^% DoNotWriteHistories/DoNotWriteHistories/' \
		"${SOURCE}/params.txt" >"${run}/params.txt"
	cat >>"${run}/params.txt" <<'EOF'

# Full-sky validation sphere centred in the periodic box.
PLCProvideConeData
PLCCenter 1935.0 1935.0 1935.0
PLCAxis 0.0 0.0 1.0
EOF
	awk '$1 == "0.492562563187" {keep=1} keep' "${SOURCE}/outputs" >"${run}/outputs"
	if [[ $(head -n 1 "${run}/outputs") != "0.492562563187" ]] || \
		[[ $(tail -n 1 "${run}/outputs") != "0.0" ]]; then
		echo "Failed to construct low-redshift output list for ${label}" >&2
		exit 2
	fi
done

{
	echo "staged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "source_run=${SOURCE}"
	echo "source_binary_sha256=$(sha256sum "${SOURCE}/pinocchio.x" | awk '{print $1}')"
	echo "geppetto_commit=$(git -C "${REPO}" rev-parse HEAD)"
	echo "grid_size=2160"
	echo "box_size_mpc_h=3870"
	echo "plc_aperture_deg=180"
	echo "plc_z_max=0.492562563187"
	echo "plc_center_mpc_h=1935.0,1935.0,1935.0"
	echo "array_concurrency=5"
	echo "nodes_per_pinocchio=24"
	echo "mpi_ranks_per_pinocchio=720"
	echo "cpus_per_mpi_rank=3"
	echo "seeds=${SEEDS[*]}"
} >"${BASE}/campaign_provenance.env"

cd "${BASE}"
pinocchio_job=$(sbatch --parsable --array=0-7%5 "${CAMPAIGN}/submit_pinocchio_array.sh")
geppetto_job=$(
	sbatch --parsable --array=0-7%8 --dependency="aftercorr:${pinocchio_job}" \
		"${CAMPAIGN}/submit_geppetto_array.sh"
)
measure_job=$(
	sbatch --parsable --array=0-7%8 --dependency="aftercorr:${geppetto_job}" \
		"${CAMPAIGN}/submit_measure_array.sh"
)
aggregate_job=$(
	sbatch --parsable --dependency="afterok:${measure_job}" \
		"${CAMPAIGN}/submit_aggregate.sh"
)

{
	echo "PINOCCHIO_ARRAY_JOB_ID=${pinocchio_job}"
	echo "GEPPETTO_ARRAY_JOB_ID=${geppetto_job}"
	echo "MEASURE_ARRAY_JOB_ID=${measure_job}"
	echo "AGGREGATE_JOB_ID=${aggregate_job}"
} >pipeline_jobs.env
cat pipeline_jobs.env
