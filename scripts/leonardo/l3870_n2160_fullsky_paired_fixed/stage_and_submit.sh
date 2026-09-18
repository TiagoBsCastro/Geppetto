#!/bin/bash
set -euo pipefail

SOURCE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000"
BASE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/fullsky_z0493_paired_fixed"
REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
CAMPAIGN="${REPO}/scripts/leonardo/l3870_n2160_fullsky_paired_fixed"
mapfile -t REALIZATIONS < <(awk 'NF && $1 !~ /^#/' "${CAMPAIGN}/realizations.tsv")
if ((${#REALIZATIONS[@]} != 8)); then
	echo "Expected eight paired-fixed realizations" >&2
	exit 2
fi
if [[ -e "${BASE}/pipeline_jobs.env" ]]; then
	echo "Campaign was already submitted: ${BASE}/pipeline_jobs.env" >&2
	exit 2
fi
mkdir -p "${BASE}/logs"

for realization in "${REALIZATIONS[@]}"; do
	read -r label seed paired <<<"${realization}"
	run="${BASE}/${label}"
	if compgen -G "${run}/pinocchio.000.massmap.seg*.fits" >/dev/null; then
		echo "Refusing to overwrite existing outputs in ${run}" >&2
		exit 2
	fi
	mkdir -p "${run}"
	ln -sfn "${SOURCE}/pinocchio.x" "${run}/pinocchio.x"
	ln -sfn "${SOURCE}/CambFiles" "${run}/CambFiles"
	sed \
		-e "s/^RandomSeed[[:space:]].*/RandomSeed             ${seed}/" \
		-e 's/^StartingzForPLC[[:space:]].*/StartingzForPLC        0.492562563187/' \
		-e 's/^PLCAperture[[:space:]].*/PLCAperture            180/' \
		-e 's/^MaxMem[[:space:]].*/MaxMem                14000/' \
		-e 's/^% DoNotWriteHistories/DoNotWriteHistories/' \
		-e '/^[[:space:]]*FixedIC[[:space:]]*$/d' \
		-e '/^[[:space:]]*PairedIC[[:space:]]*$/d' \
		"${SOURCE}/params.txt" >"${run}/params.txt"
	cat >>"${run}/params.txt" <<'EOF'

# Full-sky paired-and-fixed validation sphere centred in the periodic box.
PLCProvideConeData
PLCCenter 1935.0 1935.0 1935.0
PLCAxis 0.0 0.0 1.0
FixedIC
EOF
	if ((paired)); then
		echo "PairedIC" >>"${run}/params.txt"
	fi
	awk '$1 == "0.492562563187" {keep=1} keep' "${SOURCE}/outputs" >"${run}/outputs"
done

{
	echo "staged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "source_binary_sha256=$(sha256sum "${SOURCE}/pinocchio.x" | awk '{print $1}')"
	echo "geppetto_commit=$(git -C "${REPO}" rev-parse HEAD)"
	echo "fixed_ic=1"
	echo "paired_pairs=4"
	printf 'realization=%s\n' "${REALIZATIONS[@]}"
} >"${BASE}/campaign_provenance.env"

cd "${BASE}"
pinocchio_job=$(sbatch --parsable --array=0-7%5 "${CAMPAIGN}/submit_pinocchio_array.sh")
geppetto_job=$(sbatch --parsable --array=0-7%8 --dependency="aftercorr:${pinocchio_job}" "${CAMPAIGN}/submit_geppetto_array.sh")
measure_job=$(sbatch --parsable --array=0-7%8 --dependency="aftercorr:${geppetto_job}" "${CAMPAIGN}/submit_measure_array.sh")
{
	echo "PINOCCHIO_ARRAY_JOB_ID=${pinocchio_job}"
	echo "GEPPETTO_ARRAY_JOB_ID=${geppetto_job}"
	echo "MEASURE_ARRAY_JOB_ID=${measure_job}"
} >pipeline_jobs.env
cat pipeline_jobs.env
